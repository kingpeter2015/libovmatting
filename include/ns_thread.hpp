#ifndef __NS__THREAD__HPP___
#define __NS__THREAD__HPP___

#include <thread>
#include <atomic>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>

class ov_simple_thread
{
public:
	ov_simple_thread();
	virtual ~ov_simple_thread();

protected:
	void start();
	std::thread::id getId();
	void interrupt();
	bool isInterrupted();
	void join();
	virtual void run();

private:
	std::atomic<bool> _interript = false;
	std::thread _thread;
};


#endif