#include <iostream>

namespace mancalaCuda
{
	constexpr int nPits_player = 3;
	constexpr int nPits_total = (nPits_player + 1) * 2;
	constexpr int nSeeds = 3;
	
	union board_state
	{
		int pits[nPits_total];
		struct
		{
		int player1pits[nPits_player];
		int player1pool;
		int player2pits[nPits_player];
		int player2pool;
		};
	};

    struct turn_record
    {
		board_state state;
        int action;
        int reward;
    };



	class RLagent
	{
		private:
		std::string name;
		public:
		RLagent();
		
		std::string GetName();
	};
}