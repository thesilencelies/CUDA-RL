#include "rlagent.hpp"

namespace mancalaCuda
{
	RLagent::RLagent()
	{
		name = "rlagent";
	}
	
	std::string RLagent::GetName()
	{
		return name;
	}
}