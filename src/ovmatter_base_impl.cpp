#include "ovmatter_base_impl.hpp"

using namespace ovlib::matter;

std::map<std::string, NewInstancePt> CMatterFactory::dynCreateMap;
std::map<int, NewInstancePt> CMatterFactory::dynCreateMapType;

double MatterBaseImpl::getAttributeValue(std::string attName)
{
    if(attName == "InferCount")
    {
        return m_nInferCount;
    }
    else if(attName == "Interval")
    {
        return m_nInterval;
    }
    else if(attName == "MotionThreshold")
    {
        return m_fMotionThreshold;
    }
    else if(attName == "ForceInferLimit")
    {
        return m_nForceInferLimit;
    }
    else
    {
        return -1.0;
    }
}

MatterChannel* MatterBaseImpl::setAttributeValue(std::string attrName, double dblValue)
{
    if(attrName == "InferCount")
    {
        m_nInferCount = (int)dblValue;
    }
    else if(attrName == "Interval")
    {
        m_nInterval = (int)dblValue;
    }
    else if(attrName == "MotionThreshold")
    {
        m_fMotionThreshold = dblValue;
    }
    else if(attrName == "ForceInferLimit")
    {
        m_nForceInferLimit = (int)dblValue;
    }

    return this;
}