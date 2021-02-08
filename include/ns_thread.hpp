#ifndef __NS__THREAD__HPP___
#define __NS__THREAD__HPP___

#include <thread>
#include <atomic>
#include <iostream>
#include <functional>
#include <vector>
#include <queue>
#include <list>
#include <algorithm>
#include <mutex>
#include <condition_variable>

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
	std::atomic<bool> _interript;
	std::thread _thread;
};

template <typename T>
class ConcurrentQueue
{
	std::queue<T> queue_;
	mutable std::mutex mutex_;

	// Moved out of public interface to prevent races between this
	// and pop().
	bool empty() const
	{
		return queue_.empty();
	}

public:
	ConcurrentQueue() = default;
	ConcurrentQueue(const ConcurrentQueue<T>&) = delete;
	virtual ~ConcurrentQueue() {}
	ConcurrentQueue(ConcurrentQueue<T>&& other)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_ = std::move(other.queue_);
	}

	ConcurrentQueue& operator=(const ConcurrentQueue<T>&) = delete;

	unsigned long size() const
	{
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.size();
	}

	T pop()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty())
		{
			return {};
		}
		T tmp = queue_.front();
		queue_.pop();
		return tmp;
	}

	void pop(T& item)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty())
		{
			return;
		}
		item = queue_.front();
		queue_.pop();
	}

	void front(T& item)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty())
		{
			return;
		}
		item = queue_.front();
	}

	void popAll(std::vector<T>& items)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		while (!queue_.empty())
		{
			T item = queue_.front();
			queue_.pop();
			items.push_back(item);
		}
	}

	void push(const T& item)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(item);
	}

	void clear()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		while (!queue_.empty())
		{
			queue_.pop();
		}
	}

};


#endif