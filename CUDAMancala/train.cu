#include <iostream>
#include "rlagent.cuh"

using namespace mancalaCuda;
int main(void)
{
	RLAgent testAgent;

	std::cout << "running agent : " << testAgent.GetName() << std::endl;
	testAgent.RunStep();
	std::cout << "run1" << std::endl << testAgent.PrintRun();
	testAgent.RunStep();
	std::cout << "run2" << std::endl << testAgent.PrintRun();
}