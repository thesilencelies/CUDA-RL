#include "rlagent.cuh"
#include <iostream>

using namespace mancalaCuda;
int main(void)
{
	RLAgent testAgent;

	//reduction test
	std::vector<int> reduceArray;
	for(int i = 0; i < 10000; i++)
	{
		reduceArray.push_back(i);
	}
	std::cout << "max value was : " << testAgent.GetMax(reduceArray) << std::endl;

	testAgent.LoadQMat("C:/Users/stephen/Documents/GitHub/CUDA-RL/AIData/Tabular3_1.pb");

	std::cout << "running agent : " << testAgent.GetName() << std::endl;
	testAgent.RunStep();
	//std::cout << "run1" << std::endl << testAgent.PrintRun();
	testAgent.RunStep();
	//std::cout << "run2" << std::endl << testAgent.PrintRun();

	testAgent.SaveQMat("C:/Users/stephen/Documents/GitHub/CUDA-RL/AIData/Tabular3_3.pb");
}