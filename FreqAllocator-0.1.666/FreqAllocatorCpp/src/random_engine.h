#include <random>

struct RandomEngine
{
	std::default_random_engine rng;
	RandomEngine()
		: rng(std::random_device()())
	{		
	}

	void seed(unsigned int x)
	{
		rng.seed(x);
	}

	static std::default_random_engine& get_instance()
	{
		static RandomEngine inst;
		return inst.rng;
	}

};