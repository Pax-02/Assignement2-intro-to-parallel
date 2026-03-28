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

//function to run local static work after each rank gets it's block

double run_static_local(int rank, int size, int K, int func_id, double tol,long long *local_accepted, int *local_coarse_intervals){
    //minimum number each rank will get
    int base  = K / size;
    int rem   = K % size;
    //add the reminder incase there is one by one
    int count = base + (rank < rem ? 1 : 0);
    int start = rank * base + (rank < rem ? rank : rem);

    double width = 1.0 / (double)K;
    double local_result = 0.0;

    *local_accepted = 0;
    *local_coarse_intervals = count;

    for (int i = 0; i < count; i++) {
        int idx = start + i;
        double a = idx * width;
        double b = (idx + 1) * width;
        //divide the total tolerance across K coarse intervals for it to not exceed
        local_result += adaptive_simpson(a, b, tol / (double)K, func_id,local_accepted, 0);
    }

    return local_result;
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

    //chekc if the mode is in the range of 0 to 2
    if (mode < 0 || mode > 2) {
        if (rank == 0) {
            fprintf(stderr, "Mode must be 0, 1, or 2\n");
        }
        MPI_Finalize();
        return 1;
    }
    // serial baseline
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

    }
    else if (mode == 1){
        
        //decided to make k = size 
        int K = size;

        long long local_accepted = 0;
        long long global_accepted = 0;

        int local_coarse_intervals = 0;

        double local_result = 0.0;
        double global_result = 0.0;

        double local_compute_time = 0.0;
        double global_runtime = 0.0;

        long long *all_accepted = NULL;
        int *all_coarse = NULL;
        double *all_times = NULL;

        //use rank 0 to allocate memory
        if (rank == 0) {
            all_accepted = (long long *)malloc(size * sizeof(long long));
            all_coarse   = (int *)malloc(size * sizeof(int));
            all_times    = (double *)malloc(size * sizeof(double));

            if (all_accepted == NULL || all_coarse == NULL || all_times == NULL) {
                fprintf(stderr, "Issue with memory allocation on rank 0.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        //wait till rank 0 is done
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        //each rank computes it's chunck
        local_result = run_static_local(rank, size, K, func_id, tol, &local_accepted, &local_coarse_intervals);

        local_compute_time = MPI_Wtime() - t0;

        MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_accepted, &global_accepted, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Gather(&local_accepted, 1, MPI_LONG_LONG, all_accepted, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_coarse_intervals, 1, MPI_INT, all_coarse, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_compute_time, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double local_total_time = MPI_Wtime() - t0;
        MPI_Reduce(&local_total_time, &global_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        //use rank 0 to consolidate 
        if (rank == 0) {

            double max_time = all_times[0];
            double sum_time = 0.0;

            for (int r = 0; r < size; r++) {
                if (all_times[r] > max_time) {
                    max_time = all_times[r];
                }
                sum_time += all_times[r];
            }

            double avg_time = sum_time / (double)size;
            double imbalance_ratio = (avg_time > 0.0) ? (max_time / avg_time) : 0.0;

            printf("Mode: %d (MPI Static Decomposition)\n", mode);
            printf("Processes: %d\n", size);
            printf("Function ID: %d\n", func_id);
            printf("Tolerance: %.12e\n", tol);
            printf("Chosen K: %d\n", K);
            printf("Integral estimate: %.15f\n", global_result);
            printf("Accepted intervals: %lld\n", global_accepted);
            printf("Runtime in seconds: %.6f\n", global_runtime);

            printf("Per-rank coarse intervals and work:\n");
            for (int r = 0; r < size; r++) {
                printf("  Rank %d -> coarse intervals: %d, accepted intervals: %lld, compute time (s): %.6f\n",
                       r, all_coarse[r], all_accepted[r], all_times[r]);
            }

            printf("Observed load imbalance (max_time / avg_time): %.6f\n", imbalance_ratio);

            free(all_accepted);
            free(all_coarse);
            free(all_times);
        }
    }

    MPI_Finalize();
    return 0;

}