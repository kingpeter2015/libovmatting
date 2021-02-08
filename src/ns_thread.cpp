#include "ns_thread.hpp"


ov_simple_thread::ov_simple_thread()
{
	_interript = false;
}


ov_simple_thread::~ov_simple_thread()
{
	if (!this->isInterrupted())
	{
		this->interrupt();
	}

	if (_thread.joinable()) {
		_thread.join();
	}
}

void ov_simple_thread::start()
{
	_interript = false;
	std::thread thr(std::bind(&ov_simple_thread::run, this));
	_thread = std::move(thr);

}

std::thread::id ov_simple_thread::getId()
{
	return _thread.get_id();
}

void ov_simple_thread::interrupt()
{
	_interript = true;
}

bool ov_simple_thread::isInterrupted()
{
	return _interript;
}

void ov_simple_thread::join()
{
	_thread.join();
}

void ov_simple_thread::run()
{

}
