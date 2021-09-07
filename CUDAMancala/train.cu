#include <iostream>
#include "rlagent.cuh"

using namespace mancalaCuda;
int main(void)
{
	RLagent testAgent;

	std::cout << "hello world : " << testAgent.GetName() << std::endl;
	testAgent.TrainStep();
}