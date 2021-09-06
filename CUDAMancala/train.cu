#include <iostream>
#include "rlagent.hpp"

using namespace mancalaCuda;
int main(void)
{
	RLagent testAgent;

	std::cout << "hello world : " << testAgent.GetName() << std::endl;
}