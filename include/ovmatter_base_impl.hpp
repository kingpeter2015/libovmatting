#ifndef __OVMATTER__BASE_IMPL_HPP__
#define __OVMATTER__BASE_IMPL_HPP__

/**
* MIT License

Copyright (c) 2021 kingpeter2015

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

* this is base class of all the matters
**/

#include "ovmatter.h"

namespace ovlib
{
	namespace matter
	{
		class MatterBaseImpl : public MatterChannel
		{
		public:
			MatterBaseImpl() {};
			virtual ~MatterBaseImpl() {};
			virtual bool init(const MatterParams& params) = 0;
			virtual int getInferCount(){ return m_nInferCount; }

			MatterParams _params;
		
		protected:
			int m_nInterval;
			float m_fMotionThreshold;
			int m_nInferCount = 0;
			double m_preDiff = 0.0;
		};

		typedef MatterBaseImpl* (*NewInstancePt)();
		class CMatterFactory
		{
		public:
			static MatterBaseImpl* CreateMatter(const char* className)
			{
				std::map<std::string, NewInstancePt>::const_iterator it = dynCreateMap.find(className);
				if (it == dynCreateMap.end())
				{
					return NULL;
				}
				else
				{
					NewInstancePt np = it->second;
					return np();
				}
			}

			static MatterBaseImpl* CreateMatter(const int code)
			{
				std::map<int, NewInstancePt>::const_iterator it = dynCreateMapType.find(code);
				if (it == dynCreateMapType.end())
				{
					return NULL;
				}
				else
				{
					NewInstancePt np = it->second;
					return np();
				}
			}

			static void RegisterClass(int code, const char* className, NewInstancePt np)
			{
				dynCreateMap[className] = np;
				dynCreateMapType[code] = np;
			}		

		private:
			static std::map<std::string, NewInstancePt> dynCreateMap;
			static std::map<int, NewInstancePt> dynCreateMapType;
		};

		class MatterRegister
		{
		public:
			MatterRegister(int code, const char* className, NewInstancePt np)
			{
				CMatterFactory::RegisterClass(code, className, np);
			}
		};

#define REGISTER_MATTER_CLASS(code, class_name) \
class class_name##MatterRegister \
{ \
public: \
	static MatterBaseImpl* NewInstance() \
	{ \
		return new class_name(); \
	} \
private: \
	static MatterRegister reg; \
}; \
MatterRegister class_name##MatterRegister::reg(code, #class_name, class_name##MatterRegister::NewInstance);


	}; // namespace matter
}; // namespace ovlib


#endif