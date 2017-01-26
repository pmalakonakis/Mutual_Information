#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

#define THREAD_COUNT 4
#define NUM_VARS 4 // number of random variables

int M = 100000; // time series length
int resol = 2*96; // num of histogram Bins

float ret[NUM_VARS][NUM_VARS];

void equiwidthhist(float *x,float *y,int size, float *probx, float *proby, float *probx_y);

typedef struct {
	int var1;
	int var2;
	int id;
	float *x;
	float *y;
	max_engine_t *engine;
} arg_t;


void equiwidthhist(float *x,float *y,int size, float *probx, float *proby, float *probx_y){

	float maxx=0,maxy=0,minx,miny;
	int i,l,k;
	int R = size;

	miny = 0.0;
	minx = 0.0;
	maxy = y[0];
	maxx = x[0];

	for (i=1; i<M; i++){
		if (maxx<x[i])
			maxx=x[i];

		if (maxy<y[i])
			maxy=y[i];

		if (minx>x[i])
			minx=x[i];

		if (miny>y[i])
			miny=y[i];
	}

	//create histogram
	for (i=0; i<M; i++){

		k = (int)(R * (x[i]-minx)/(maxx-minx));
		if (k>=0 && k<R)
			probx[k] +=1.0;

		l = (int)(R * (y[i]-miny)/(maxy-miny));

		if (l>=0 && l<R)
			proby[l] +=1.0;

		if (k>=0 && k<R)
			if (l>=0 && l<R)
				probx_y[k+R*l]+=1.0;
	}


}

float MIsoft(float *x,float *y)
{
	const long int resoln = resol;
	long int resolnr = 0;
	resolnr=((resoln*4)+48-1)/48;
	resolnr=48*resolnr/4;
	const long int size = resolnr;

	long int R = size;

	float *probx= calloc(size,4);
	float *proby= calloc(size,4);
	float *probx_y= calloc(size*size,4);

	float logs;
	float mi=0.0;
	int i,j;

	//pdf estimation with equiwidth histogram
	equiwidthhist(x,y,size,probx, proby,probx_y);

	//calculate MI
	for (j=0; j<R; j++)
		for (i=0; i<R; i++){
			logs = log2(((float)(probx_y[i+R*j]/M)) /(((float)probx[i]/M) * ((float)proby[j]/M)));
			if (isfinite(logs))
				mi = mi + ((float)probx_y[i+R*j]/M) * logs;
		}

	free (probx);
	free (probx_y);
	free (proby);


	return mi;
}



void *MI(void *arg)
{
	int id = ((arg_t *)arg)->id;
	float *x = ((arg_t *)arg)->x;
	float *y = ((arg_t *)arg)->y;
	max_engine_t *engine = ((arg_t *)arg)->engine;
	int var1 = ((arg_t *)arg)->var1;
	int var2 = ((arg_t *)arg)->var2;

	const long int resoln = resol;
	long int resolnr = 0;
	resolnr=((resoln*4)+48-1)/48;
	resolnr=48*resolnr/4;

	const long int size = resolnr;
	long int  sizeBytes = size * sizeof(float);

	float *s = malloc(size*sizeBytes);
	long int R = size;

	float *probx= calloc(size,4);
	float *probx1= calloc(size*size/3,4);
	float *probx_y= calloc(size*size,4);
	float *proby= calloc(size,4);

	int i,z,j;

	//pdf estimation with equiwidth histogram
	equiwidthhist(x,y,size,probx, proby,probx_y);

	//copy probx queue_input ->high overhead
	for (i=0; i<R/3; i++){
		for (j=0; j<R; j++){
			probx1[R*i+j]=probx[j];
		}
	}

/// run on DFE
	max_file_t *maxfile = MutualInformation_init();
	max_actions_t *act = max_actions_init(maxfile, "default");

	max_set_param_uint64t(act, "N", size);
	max_set_param_uint64t(act, "Ns", size*size/3);
	max_set_param_double(act, "m",(float) M);


	max_queue_input(act, "probx_y", probx_y,size*sizeBytes/3); // multiple of 16

	max_queue_input(act, "probx_y2", &probx_y[R/3*R],size* sizeBytes/3); // multiple of 16

	max_queue_input(act, "probx_y3", &probx_y[R*2*R/3],size* sizeBytes/3); // multiple of 16

	max_queue_input(act, "probx", probx1, size* sizeBytes/3);

	max_queue_input(act, "proby", proby, sizeBytes/3);
	max_queue_input(act, "proby2", &proby[R/3], sizeBytes/3);
	max_queue_input(act, "proby3", &proby[2*R/3], sizeBytes/3);

	max_queue_output(act, "s", s,size * sizeBytes/3);


	printf("Running on DFE%d (mode: ComputeWithScalar)...\n",id);
	max_run(engine,act);

		//calculate TE (Sum)
	float mi = 0.0;
	for (z=R-13; z<R; z++){
		mi= mi + s[R*((R/3)-1)+z];
	}

	free (probx);
	free (probx_y);
	free (proby);
	free (s);

	ret[var1][var2] = mi;
	pthread_exit(NULL);
	return NULL;
}




int main(){

	struct timeval start, stop;
	pthread_t threads[THREAD_COUNT];
	float  a = 11.0;
	int i,j,h;

	float **rvar =(float **)malloc(NUM_VARS * sizeof(float *));
	for (i=0; i<NUM_VARS; i++)
		rvar[i] = (float *)malloc(M * sizeof(float));

	float **misft =(float **)malloc(NUM_VARS * sizeof(float *));
	for (i=0; i<NUM_VARS; i++)
		misft[i] = (float *)malloc(NUM_VARS * sizeof(float));

	max_file_t *maxfile = MutualInformation_init();

	max_engine_t *engine1=max_load(maxfile, "*");
	max_engine_t *engine2=max_load(maxfile, "*");
	max_engine_t *engine3=max_load(maxfile, "*");
	max_engine_t *engine4=max_load(maxfile, "*");

	//generate input
	for (j=0; j<NUM_VARS; j++)
		for (i=0; i<M; i++){
			rvar[j][i]= ((float)rand()/(float)(RAND_MAX/a));
		}
/////////////////////////////////////////////////////////////////////////////////////////
///software call
///////////////////////////////////////////////////////////////////////////////
	gettimeofday(&start, NULL);
	for (j=0; j<NUM_VARS; j++)
		for (h=j+1; h<NUM_VARS; h++){
			misft[j][h]=MIsoft(rvar[j],rvar[h]);
		}
	gettimeofday(&stop, NULL);
	printf("END - Time Software:    %ld μs\n", ((stop.tv_sec * 1000000 + stop.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec)));
	printf("Done.\n");


////////////////////////////////////////////////////////////////////////////
///hardware call
///////////////////////////////////////////////////////////////////////////////
	gettimeofday(&start, NULL);
	arg_t arg[4];
	arg[0].engine=engine1;
	arg[1].engine=engine2;
	arg[2].engine=engine3;
	arg[3].engine=engine4;
	arg[0].id=0;
	arg[1].id=1;
	arg[2].id=2;
	arg[3].id=3;

	int threadsstart=0;
	j=0;
	h=1;
	for (int k=0; k<(NUM_VARS*(NUM_VARS-1)/2); k+=THREAD_COUNT){
		for (i = 0; i < THREAD_COUNT; i++){
			if (h>=NUM_VARS){
				j=j+1;
				h=j+1;
				if (j>=NUM_VARS||h>=NUM_VARS)
						break;
			}
			arg[i].x=rvar[j];
			arg[i].y=rvar[h];
			arg[i].var1=j;
			arg[i].var2=h;
			pthread_create(&threads[i], NULL, &MI, (void*)&arg[i]);
			threadsstart++;
			h=h+1;
		}
			for (int i = 0; i < threadsstart; ++i)
				pthread_join(threads[i], NULL);
			threadsstart=0;
	}

	max_unload(engine1);
	max_unload(engine2);
	max_unload(engine3);
	max_unload(engine4);
	gettimeofday(&stop, NULL);
	printf("END - Time DFE:    %ld μs\n", ((stop.tv_sec * 1000000 + stop.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec)));
	printf("Done.\n");
	printf("MI Results \n");
	printf("Hardware \tSoftware \n");
	for (i=0;i<NUM_VARS;i++)
		for(j=0;j<NUM_VARS;j++)
			printf("%f\t%f\n",ret[i][j],misft[i][j]);

	return 0 ;
}
