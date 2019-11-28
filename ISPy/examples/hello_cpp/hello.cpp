#include <iostream>
#include <cstring>
#include "hello.h"

using namespace std;

namespace greetings {
	Hello::Hello() {
		char str[13]   = "Hello world!";
		Hello::message = new char[13];
		strcpy(Hello::message, str);
	}

	Hello::Hello(Hello const& copy) {
		Hello::message = new char[13];
		strcpy(Hello::message, copy.message);
	}

	Hello& Hello::operator=(Hello const& copy) {
		strcpy(Hello::message, copy.message);

		return *this;
	}

	/* Extreme care is required here:
	 *
	 * Do not implement the destructor without properly implementing the copy
	 * and the assignment operators properly, or you might end up with double
	 * free!!
	 */
	Hello::~Hello() {
		delete Hello::message;
	}

	void Hello::printMessage(void) {
		cout << Hello::message << endl;
	}
}

int main(int argc, char* argv[]) {
	greetings::Hello* helloInstance = new greetings::Hello();
	helloInstance->printMessage();
	delete helloInstance;

	return 0;
}
