# include <chrono>
# include <random>
# include <cstdlib>
# include <sstream>
# include <string>
# include <fstream>
# include <iostream>
# include <iomanip>
# include <mpi.h>

// Attention , ne marche qu'en C++ 11 ou supérieur :
double approximate_pi( unsigned long nbSamples ) 
{
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = beginning.time_since_epoch();
    unsigned seed = d.count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution <double> distribution ( -1.0 ,1.0);
    unsigned long nbDarts = 0;
    // Throw nbSamples darts in the unit square [-1 :1] x [-1 :1]
    for ( unsigned sample = 0 ; sample < nbSamples ; ++ sample ) {
        double x = distribution(generator);
        double y = distribution(generator);
        // Test if the dart is in the unit disk
        if ( x*x+y*y<=1 ) nbDarts ++;
    }
    // Number of nbDarts throwed in the unit disk
    double ratio = double(nbDarts)/double(nbSamples);
    return 4*ratio;
}

int main( int nargs, char* argv[] )
{
	// On initialise le contexte MPI qui va s'occuper :
	//    1. Créer un communicateur global, COMM_WORLD qui permet de gérer
	//       et assurer la cohésion de l'ensemble des processus créés par MPI;
	//    2. d'attribuer à chaque processus un identifiant ( entier ) unique pour
	//       le communicateur COMM_WORLD
	//    3. etc...
	MPI_Init( &nargs, &argv );
	// Pour des raisons de portabilité qui débordent largement du cadre
	// de ce cours, on préfère toujours cloner le communicateur global
	// MPI_COMM_WORLD qui gère l'ensemble des processus lancés par MPI.
	MPI_Comm globComm;
	MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
	// On interroge le communicateur global pour connaître le nombre de processus
	// qui ont été lancés par l'utilisateur :
	int nbp;
	MPI_Comm_size(globComm, &nbp);
	// On interroge le communicateur global pour connaître l'identifiant qui
	// m'a été attribué ( en tant que processus ). Cet identifiant est compris
	// entre 0 et nbp-1 ( nbp étant le nombre de processus qui ont été lancés par
	// l'utilisateur )
	int rank;
	MPI_Comm_rank(globComm, &rank);
	// Création d'un fichier pour ma propre sortie en écriture :
	std::stringstream fileName;
	fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
	std::ofstream output( fileName.str().c_str() );

	unsigned long nbSamples = 10000000UL;
	if (nargs > 1) {
		nbSamples = std::strtoul(argv[1], nullptr, 10);
	}

	// Répartition des échantillons
	unsigned long base = nbSamples / nbp;
	unsigned long rem = nbSamples % nbp;
	unsigned long localSamples = base + (static_cast<unsigned long>(rank) < rem ? 1UL : 0UL);

	MPI_Barrier(globComm);
	double t0 = MPI_Wtime();

	// Calcul local
	typedef std::chrono::high_resolution_clock myclock;
	myclock::time_point beginning = myclock::now();
	myclock::duration d = beginning.time_since_epoch();
	unsigned seed = static_cast<unsigned>(d.count()) + static_cast<unsigned>(rank * 12345);
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	unsigned long localHits = 0;
	for (unsigned long sample = 0; sample < localSamples; ++sample) {
		double x = distribution(generator);
		double y = distribution(generator);
		if (x * x + y * y <= 1.0) localHits++;
	}

	unsigned long globalHits = 0;
	unsigned long globalSamples = 0;
	MPI_Reduce(&localHits, &globalHits, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, globComm);
	MPI_Reduce(&localSamples, &globalSamples, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, globComm);

	MPI_Barrier(globComm);
	double t1 = MPI_Wtime();

	if (rank == 0) {
		double pi = 4.0 * static_cast<double>(globalHits) / static_cast<double>(globalSamples);
		output << "nbSamples=" << globalSamples << "\n";
		output << "pi≈" << std::setprecision(10) << pi << "\n";
		output << "time=" << (t1 - t0) << " s\n";
		std::cout << "Temps (MPI) : " << (t1 - t0) << " s, pi≈" << std::setprecision(10) << pi << "\n";
	}

	output.close();
	// A la fin du programme, on doit synchroniser une dernière fois tous les processus
	// afin qu'aucun processus ne se termine pendant que d'autres processus continue à
	// tourner. Si on oublie cet instruction, on aura une plantage assuré des processus
	// qui ne seront pas encore terminés.
	MPI_Finalize();
	return EXIT_SUCCESS;
}

