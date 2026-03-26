#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
//set the safety stop for the divison to 50
#define MAX_DEPTH 50

// Function to determine wich function to use depending on the input

double f(double x, int func_id){
    switch (func_id) {
        case 0:
            return sin(x) + 0.5 * cos(3.0 * x);
        case 1: {
            double t = x - 0.3;
            return 1.0 / (1.0 + 100.0 * t * t);
        }
        case 2:
            return sin(200.0 * x) * exp(-x);
        default:
            fprintf(stderr, "Invalid func_id: %d (it must be 0, 1, or 2)\n", func_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 0.0;
    }
}

//simpson estimate function on [a,b]

double simpson_estimate(double a, double b, int func_id) {
    double m = 0.5 * (a + b);
    double fa = f(a, func_id);
    double fm = f(m, func_id);
    double fb = f(b, func_id);

    return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
}

// Recursive adaptive simpson 

double adaptive_simpson(double a, double b, double tol, int func_id,long long *accepted_intervals, int depth) {

    //let m
    double m = 0.5 * (a + b);

    double s_whole = simpson_estimate(a, b, func_id);
    double s_left  = simpson_estimate(a, m, func_id);
    double s_right = simpson_estimate(m, b, func_id);
    double s_refined = s_left + s_right;
    //calculate error
    double err = fabs(s_refined - s_whole);

    //check if we have reached to the safety stop
    if (depth >= MAX_DEPTH) {

        (*accepted_intervals)++;
        return s_refined;
    }

    //check if the error is in the range of tol

    if (err <= tol) {
        (*accepted_intervals)++;
        return s_refined;
    }

    //Else we split it again
    return adaptive_simpson(a, m, tol / 2.0, func_id, accepted_intervals, depth + 1)
         + adaptive_simpson(m, b, tol / 2.0, func_id, accepted_intervals, depth + 1);
}

//serial basiline for[0,1]

double run_serial(int func_id, double tol, long long *accepted_intervals) {
    *accepted_intervals = 0;
    return adaptive_simpson(0.0, 1.0, tol, func_id, accepted_intervals, 0);
}

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //check the arguments we have 
    if (argc != 4) {
        if (rank == 0) {

            fprintf(stderr, "Cpmmand should be: mpirun -np P ./integration func_id mode tol\n Example Example: mpirun -np 1 ./integration 1 0 1e-8\n");
        }

        MPI_Finalize();
        return 1;
    }
    
    int func_id = atoi(argv[1]);
    int mode    = atoi(argv[2]);
    double tol  = atof(argv[3]);

    //check the tolerance if it's not negative number
    if (tol <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Tolerance must be greater than or equak to 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    //chekc if the func_id is in the range of 0 to 2
    if (func_id < 0 || func_id > 2) {
        if (rank == 0) {
            fprintf(stderr, "func_id must be 0, 1, or 2\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (mode == 0){
        if (rank == 0) {
        long long accepted_intervals = 0;

        double t0 = MPI_Wtime();
        double result = run_serial(func_id, tol, &accepted_intervals);
        double t1 = MPI_Wtime();

        printf("Mode: %d \n", mode);
        printf("Processes: %d\n", size);
        printf("Function ID: %d\n", func_id);
        printf("Tolerance: %.12e\n", tol);
        printf("Integral estimate: %.15f\n", result);
        printf("Accepted intervals: %lld\n", accepted_intervals);
        printf("Runtime in sec: %.6f\n", t1 - t0);
    }

    MPI_Finalize();
    return 0;
    
    }

}