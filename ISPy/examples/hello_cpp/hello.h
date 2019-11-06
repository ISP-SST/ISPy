#ifndef HELLO_H
#define HELLO_H

namespace greetings {
	class Hello {
		public:
			char *message;
			Hello();
			Hello(Hello const&);
			Hello& operator=(Hello const&);
			~Hello();
			void printMessage(void);
	};
}

#endif
