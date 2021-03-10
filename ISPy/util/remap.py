import numpy as np
from astropy.io import fits
import astropy.units as u
from ISPy.img import interpolate2d

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sphere2img(lat, lon, latc, lonc, xcenter, ycenter, rsun, peff):
    """Conversion between Heliographic coordinates to CCD coordinates.
    Ported from sphere2img written in IDL : Adapted from Cartography.c by Rick Bogart, 
    by Xudong Sun [Eq 5&6 in https://arxiv.org/pdf/1309.2392.pdf]

    Parameters
    ----------
    lat, lon : array, array
        input heliographic coordinates (latitude and longitude)
    latc, lonc : float, float
        Heliographic longitud and latitude of the refenrence (center) pixel
    xcenter, ycenter : float, float
        Center coordinates in the image
    rsun : float
        Solar radius in pixels
    peff : float
        p-angle: the position angle between the geocentric north pole and the solar 
        rotational north pole measured eastward from geocentric north.

    Returns
    -------
    array
        Latitude and longitude in the new CCD coordinate system.
    """
    sin_asd = 0.004660
    cos_asd = 0.99998914
    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0

    if (latc != last_latc):
        sin_latc = np.sin(latc)
        cos_latc = np.cos(latc)
        last_latc = latc

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)

    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    r = rsun * cos_asd / (1.0 - cos_cang * sin_asd)
    xr = r * cos_lat * np.sin(lon - lonc)
    yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon)

    cospa = np.cos(peff)
    sinpa = np.sin(peff)
    xi = xr * cospa - yr * sinpa
    eta = xr * sinpa + yr * cospa

    xi = xi + xcenter
    eta = eta + ycenter

    return xi, eta



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def vector_transformation(peff,latitude_out,longitude_out, B0,field_x_cea,
        field_y_cea, field_z_cea, lat_in_rad=False):
    """
    Magnetic field transformation matrix (see Allen Gary & Hagyard 1990)
    [Eq 7 in https://arxiv.org/pdf/1309.2392.pdf]
    """

    nlat_out = len(latitude_out)
    nlon_out = len(longitude_out)

    PP = peff
    if lat_in_rad is False:
        BB = latitude_out[None, 0:nlat_out] * np.pi / 180.0
        LL = longitude_out[0:nlon_out, None] * np.pi / 180.0
    else:
        BB = latitude_out
        LL = longitude_out
    L0 = 0.0 # We use central meridian
    Ldif = LL - L0

    a11 = -np.sin(B0)*np.sin(PP)*np.sin(Ldif)+np.cos(PP)*np.cos(Ldif)
    a12 = np.sin(B0)*np.cos(PP)*np.sin(Ldif)+np.sin(PP)*np.cos(Ldif)
    a13 = -np.cos(B0)*np.sin(Ldif)
    a21 = -np.sin(BB)*(np.sin(B0)*np.sin(PP)*np.cos(Ldif)+np.cos(PP)*np.sin(Ldif))-np.cos(BB)*np.cos(B0)*np.sin(PP)
    a22 = np.sin(BB)*(np.sin(B0)*np.cos(PP)*np.cos(Ldif)-np.sin(PP)*np.sin(Ldif))+np.cos(BB)*np.cos(B0)*np.cos(PP)
    a23 = -np.cos(B0)*np.sin(BB)*np.cos(Ldif)+np.sin(B0)*np.cos(BB)
    a31 = np.cos(BB)*(np.sin(B0)*np.sin(PP)*np.cos(Ldif)+np.cos(PP)*np.sin(Ldif))-np.sin(BB)*np.cos(B0)*np.sin(PP)
    a32 = -np.cos(BB)*(np.sin(B0)*np.cos(PP)*np.cos(Ldif)-np.sin(PP)*np.sin(Ldif))+np.sin(BB)*np.cos(B0)*np.cos(PP)
    a33 = np.cos(BB)*np.cos(B0)*np.cos(Ldif)+np.sin(BB)*np.sin(B0)

    field_x_h = a11 * field_x_cea + a12 * field_y_cea + a13 * field_z_cea
    field_y_h = a21 * field_x_cea + a22 * field_y_cea + a23 * field_z_cea
    field_z_h = a31 * field_x_cea + a32 * field_y_cea + a33 * field_z_cea

    # field_z_h positive towards earth
    # field_y_h positive towards south (-field_y_h = Bt_cea)
    # field_x_h positive towards west

    field_y_h *= -1.0

    return field_x_h, field_y_h, field_z_h





# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def remap2cea(dict_header, field_x, field_y, field_z, deltal = 0.03):
    """Map projection of the original input into the cylindical equal area system (CEA).

    Parameters
    ----------
    dict_header : dictionary
        Header with information of the observation. It works with a SDO header
        or it can be created from other data. It should include:
        dict_header = {'CRLT_OBS':float, 'RSUN_OBS':float, 'crota2':float, 'CDELT1':float, 
        'crpix1':float, 'crpix2':float, 'LATDTMAX':float, 'LATDTMAX':float 'LATDTMAX':float,
        'LONDTMAX':float, 'LATDTMIN':float, 'LONDTMIN':float, 'naxis1':float, 'naxis2':float}
        They should follow the same definition as given for SDO data:
        https://www.lmsal.com/sdodocs/doc?cmd=dcur&proj_num=SDOD0019&file_type=pdf

    field_x, field_y, field_z: array
        2D array with the magnetic field in cartesian coordinates
    deltal : float
        Heliographic degrees in the rotated coordinate system. SHARP CEA pixels are 0.03

    Returns
    -------
    array
        Remaping of the magnetic field to the cylindical equal area system (CEA).

    :Authors: 
        Carlos Diaz (ISP/SU 2020), Gregal Vissers (ISP/SU 2020)

    """
    # Latitude at disk center [rad]
    latc = dict_header['CRLT_OBS']*np.pi/180.
    B0 = np.copy(latc)
    # Longitude at disk center [rad]. The output is in Carrington coordinates. We use central meridian
    lonc = 0.0
    L0 = 0.0
    rsun = dict_header['RSUN_OBS']

    # Position angle of rotation axis
    peff = -1.0*dict_header['crota2'] * np.pi / 180.0
    dx_arcsec = dict_header['CDELT1']
    rsun_px = rsun / dx_arcsec

    # Plate locations of the image center, in units of the image radius, 
    # and measured from the corner. The SHARP CEA pixels have a linear dimension 
    # in the x-direction of 0.03 heliographic degrees in the rotated coordinate system
    dl = deltal
    xcenter = dict_header['crpix1']-1 # FITS start indexing at 1, not 0
    ycenter = dict_header['crpix2']-1 # FITS start indexing at 1, not 0
    nlat_out = np.round((dict_header['LATDTMAX'] - dict_header['LATDTMIN'])/dl)
    nlon_out = np.round((dict_header['LONDTMAX'] - dict_header['LONDTMIN'])/dl) # lon_max => LONDTMAX
    nrebin = 1
    nlat_out = int(np.round(nlat_out/nrebin)*nrebin)
    nlon_out = int(np.round(nlon_out/nrebin)*nrebin)
    nx_out = nlon_out
    ny_out = nlat_out

    latitude_out = np.arange(nlat_out)*dl + dict_header['LATDTMIN']
    longitude_out = np.arange(nlon_out)*dl + dict_header['LONDTMIN']
    lon_out = longitude_out[:, None] * np.pi / 180.0
    lat_out = lonc + latitude_out[None, :] * np.pi / 180.0

    lon_center = (dict_header['LONDTMAX'] + dict_header['LONDTMIN']) / 2. * np.pi/180.0
    lat_center = (dict_header['LATDTMAX'] + dict_header['LATDTMIN']) / 2. * np.pi/180.0
    # print(lat_cesnter,latitude_out,f[1].header['LATDTMIN'],f[1].header['LATDTMAX'])

    x_out = (np.arange(nx_out)-(nx_out-1)/2.)*dl 
    y_out = (np.arange(ny_out)-(ny_out-1)/2.)*dl 

    x_it = x_out[:,None] * np.pi/180.0
    y_it = y_out[None,:] * np.pi/180.0

    # sswidl plane2sphere equations for lat and lon; in sswidl these are fed
    # into sphere2img
    lat_it = np.arcsin(np.cos(lat_center)*y_it + np.sin(lat_center)*np.sqrt(1.0-y_it**2)*np.cos(x_it))
    lon_it = np.arcsin((np.sqrt(1.0-y_it**2)*np.sin(x_it))/np.cos(lat_it)) + lon_center

    # Heliographic coordinate to CCD coordinate
    xi, eta = sphere2img(lat_it, lon_it, latc, lonc, xcenter, ycenter, rsun_px, peff)
    
    x = np.arange(dict_header['naxis1'])
    y = np.arange(dict_header['naxis2'])

    # Interpolation (or sampling)
    xi_eta = np.concatenate([xi.flatten()[:, None], eta.flatten()[:, None]], axis=1)
    
    field_x_int = interpolate2d.interpolate2d(x, y, field_x, xi_eta).reshape((nlon_out,nlat_out))
    field_y_int = interpolate2d.interpolate2d(x, y, field_y, xi_eta).reshape((nlon_out,nlat_out))
    field_z_int = interpolate2d.interpolate2d(x, y, field_z, xi_eta).reshape((nlon_out,nlat_out))
    

    return peff, lat_it, lon_it,latc, field_x_int, field_y_int, field_z_int
    


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def bvec2cea(dict_header, field_x, field_y, field_z):
    """Transformation to Cylindrical equal area projection (CEA) from CCD
    detector as it is donde with SHARPs according to Xudong Sun (2018).

    Parameters
    ----------
    field_x, field_y, field_z: array
        2D array with the magnetic field in cartesian coordinates
    dict_header : dictionary
        Header with information of the observation. It works with a SDO header
        or it can be created from other data. It should include:
        dict_header = {'CRLT_OBS':float, 'RSUN_OBS':float, 'crota2':float, 'CDELT1':float, 
        'crpix1':float, 'crpix2':float, 'LATDTMAX':float, 'LATDTMAX':float 'LATDTMAX':float,
        'LONDTMAX':float, 'LATDTMIN':float, 'LONDTMIN':float, 'naxis1':float, 'naxis2':float}
        They should follow the same definition as given for SDO data:
        https://www.lmsal.com/sdodocs/doc?cmd=dcur&proj_num=SDOD0019&file_type=pdf


    Returns
    -------
    arrays
        Three components of the magnetic field in heliocentric spherical coordinates and
        in cylindrical equal are projection.
    
    Example
    -------
    >>> field_z = field * np.cos(inclination * np.pi / 180.0)
    >>> field_hor = field * np.sin(inclination * np.pi / 180.0)
    >>> field_y = field_hor * np.cos(azimuth * np.pi / 180.0)
    >>> field_x = -field_hor * np.sin(azimuth * np.pi / 180.0)

    >>> field_x_h2, field_y_h2, field_z_h2 = bvec2cea(file.header, field_x, field_y, field_z) 

    :Authors: 
        Carlos Diaz (ISP/SU 2020), Gregal Vissers (ISP/SU 2020)

    """

    # Map projetion
    peff, lat_it, lon_it,latc, field_x_int, field_y_int, field_z_int = remap2cea(dict_header, field_x, field_y, field_z)
    # Vector transformation
    field_x_h, field_y_h, field_z_h = vector_transformation(peff,lat_it,lon_it,latc, field_x_int, field_y_int, field_z_int, lat_in_rad=True)

    return field_x_h, field_y_h, field_z_h
