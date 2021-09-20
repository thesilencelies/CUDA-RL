#include <iostream>
#include "rlagent.cuh"

using namespace mancalaCuda;
int main(void)
{
	RLagent testAgent;

	std::cout << "running agent : " << testAgent.GetName() << std::endl;
	testAgent.TrainStep();
}