#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

int    p_verbose;
FILENAME p_blosum_qij;
FILENAME	p_db;
float  p_beta;
int    p_kmax;
int    p_kmin;
int     p_exself;
WORD	p_weights;
int	p_all_against_all;

PARAM   param[] = {
	"-v", VSWITCH	p_verbose, "Verbose mode", "0",
	"-beta", VFLOAT p_beta, "Hadamard power of blosum matrix", "0.11387",
        "-kmin", VINT p_kmin, "Min value of k for k-mers", "1",
        "-kmax", VINT p_kmax, "Max value of k for k-mers", "30",
        "-blqij", VFNAME p_blosum_qij, "Blosum qij matrix filename", "$TBCR_ALIGN/data/blosum62.qij",
	"-xs", VSWITCH  p_exself, "Exclude self", "0",
	"-w", VWORD	p_weights, "Weights on CDR [w1,w2,w3,w4,w5,w6]", "",
	"-a", VSWITCH	p_all_against_all, "Run all against all", "0",
	"-db",	VFNAME	p_db, "Receptor DB (format cdr1a cdr2a cdr3a cdr1b cdr2b cdr3b [target])", "",
	0
};

/* global variables */
float	**k1;

float	**fmatrix_k1( float **blm_qij )

{
	int	k, j;
	float	*marg;
	float	sum;

	marg = fvector( 0, 19 );
        k1 = fmatrix(0, 19, 0 , 19);

/*normalize matrix by marginal frequencies*/
        for (j=0;j<20;j++) {
              sum = 0;
              for (k=0;k<20;k++) 
                  sum+=blm_qij[j][k];
              marg[j]=sum;
        }

        /*calculate K1*/
        for (j=0;j<20;j++) {
              for (k=0;k<20;k++) {
                  k1[j][k] = blm_qij[j][k]/(marg[j]*marg[k]);
                  k1[j][k] = pow( k1[j][k], p_beta );
              }
        }

	return( k1 );
}

float    k2_prod( int *iv1, int *iv2, int start1, int start2, int k )

{
        float k2;
        int x, i1, i2;

        k2 = 1;
        for (x=0; x<k; x++) {
             i1 = iv1[x+start1];
             i2 = iv2[x+start2];
             k2 *= k1[i1][i2];
        }
        return( k2 );
}

float    k3_sum( int *iv1, int *iv2, int l1, int l2 )

{
        float k3, prod;
        int k;
        int start1,start2;

        k3 = 0;

        for (k=p_kmin; k<=p_kmax; k++) {
          for (start1=0; start1<=l1-k; start1++) {
              for (start2=0; start2<=l2-k; start2++) {
                  prod = k2_prod( iv1, iv2, start1, start2, k);
                  k3 += prod;
              }
          }
        }
        return( k3 );
}

float	**read_blosummat_qij( char *filename, char **alphabet )

{

	int	i,j, nc;
	LINELIST	*linelist, *ln;
	float	**mat;
	float	*tvec;

	linelist = linelist_read( filename );

	if ( ! linelist ) {
		printf( "Error. Cannot read linelist from %s\n", filename );
		exit( 1 );
	}

	mat = fmatrix( 0, 19, 0 ,19 );
	*alphabet = cvector( 0, 20 );

	for ( ln = linelist; ln; ln=ln->next ) {

		if ( strlen( ln->line ) <= 0 ) continue;

		if ( strncmp( ln->line, "#", 1 ) == 0 ) continue;

		if ( strncmp( ln->line, "   A", 4 ) == 0 ) {

                        for ( i=0;i<20;i++ )
                                (*alphabet)[i] = ln->line[i*7+3];

                        (*alphabet)[20] = 0;
			j=0;
                }
		else {
			tvec = fvector_split( ln->line, &nc );
			
			for ( i=0; i<nc; i++ ) {
				mat[j][i] = tvec[i];
				mat[i][j] = tvec[i];		
			}

			j++;

			fvector_free( tvec, 0, nc-1 );
		}
	}

	linelist_free( linelist );

	return( mat );
}

typedef struct pairlist2 {
        struct pairlist2 *next;
	char	*id;
        char    *cdr1a;
        char    *cdr2a;
	char    *cdr3a;
        char    *cdr1b;
	char    *cdr2b;
        char    *cdr3b;
	int	lcdr1a;
	int	lcdr2a;
	int	lcdr3a;
	int	lcdr1b;
	int	lcdr2b;
	int	lcdr3b;
	int	*ivcdr1a;
	int	*ivcdr2a;
	int	*ivcdr3a;
	int	*ivcdr1b;
	int	*ivcdr2b;
	int	*ivcdr3b;
	float	scdr1a;
	float	scdr2a;
	float	scdr3a;
	float	scdr1b;
	float	scdr2b;
	float	scdr3b;
	float	score;
	float	target;
} PAIRLIST2;

PAIRLIST2        *pairlist2_alloc()

{
        PAIRLIST2        *n;

        if ( ( n = ( PAIRLIST2 * ) malloc ( sizeof(PAIRLIST2))) != NULL ) {
		n->id = NULL;
                n->cdr1a = NULL;
                n->cdr2a = NULL;
		n->cdr3a = NULL;
		n->cdr1b = NULL;
                n->cdr2b = NULL;
		n->cdr3b = NULL;
		n->lcdr1a = -9;
		n->lcdr2a = -9;
		n->lcdr3a = -9;
		n->lcdr1b = -9;
		n->lcdr2b = -9;
		n->lcdr3b = -9;
		n->ivcdr1a = NULL;
		n->ivcdr2a = NULL;
		n->ivcdr3a = NULL;
		n->ivcdr1b = NULL;
		n->ivcdr2b = NULL;
		n->ivcdr3b = NULL;
		n->scdr1a = -99.9;
		n->scdr2a = -99.9;
		n->scdr3a = -99.9;
		n->scdr1b = -99.9;
		n->scdr2b = -99.9;
		n->scdr3b = -99.9;
		n->score = -99.9;
		n->target = -99;
                n->next = NULL;
        }

        return( n );
}

PAIRLIST2        *pairlist2_read( char *filename )

{
        PAIRLIST2        *first, *last, *new;
        FILE            *fp;
        int             ff, fc;
        LINE            cdr1a,cdr1b,cdr2a,cdr2b,cdr3a,cdr3b, id;
        LINE            line;
        int             n;
	float		target;

        first = NULL;
        n = 0;

        if ( ( fp = stream_input( filename, &fc, &ff )) == NULL ) {
                printf( "Error. Cannot read PAIRLIST2 from file %s. Exit\n",
                        filename );
                exit( 1 );
        }

        while ( fgets(line, sizeof line, fp) != NULL ) {

                if ( strncmp( line, "#", 1 ) == 0 )
                        continue;

                if ( strlen( line ) < 1 )
                        continue;

		target = -99.9;

                if ( sscanf( line, "%s %s %s %s %s %s %s %f", id, cdr1a, cdr2a, cdr3a, cdr1b, cdr2b, cdr3b, &target ) < 7 )
                        continue;

                if ( ( new = pairlist2_alloc()) == NULL ) {
                        printf( "Error. Cannot allocate pairlist. Exit\n" );
                        exit( 1 );
                }

		new->id = cvector( 0, strlen(id));
		strcpy( new->id, id );

		new->lcdr1a = strlen( cdr1a );
		new->cdr1a = cvector( 0, new->lcdr1a );
                strcpy( new->cdr1a, cdr1a );

		new->lcdr2a = strlen( cdr2a );
		new->cdr2a = cvector( 0, new->lcdr2a );	
		strcpy( new->cdr2a, cdr2a );
                
		new->lcdr3a = strlen( cdr3a );
		new->cdr3a = cvector( 0, new->lcdr3a );	
		strcpy( new->cdr3a, cdr3a );

		new->lcdr1b = strlen( cdr1b );
		new->cdr1b = cvector( 0, new->lcdr1b );	
		strcpy( new->cdr1b, cdr1b );

		new->lcdr2b = strlen( cdr2b );
		new->cdr2b = cvector( 0, new->lcdr2b );	
		strcpy( new->cdr2b, cdr2b );

		new->lcdr3b = strlen( cdr3b );
		new->cdr3b = cvector( 0, new->lcdr3b );	
		strcpy( new->cdr3b, cdr3b );

		new->target = target;

                if ( first == NULL )
                        first = new;
                else
                        last->next = new;

                last = new;
                n++;
        }

        stream_close( fp, fc, filename );

        printf( "# Read %i elements on pairlist %s\n", n, filename );

        return( first );

}

int main( int argc, char *argv[] )

{
	PAIRLIST2	 *pep1, *peplist1, *pl1;
	PAIRLIST2  *pep2, *peplist2, *pl2, *peplist_db;
	PAIRLIST2	*bestp;
	char	*alphabet;
	float	best_sco, sco_cdr1a, sco_cdr2a, sco_cdr3a, sco_cdr1b, sco_cdr2b, sco_cdr3b, sco;
	int	j, ix;
	float	**blm_qij;
	float	*w;
	int	nc;

	/* Parse command line options */
	pparse( &argc, &argv, param, 1, "tbcr_file (format ID cdr1a cdr2a cdr3a cdr1b cdr2b cdr3b [target])" );

	if ( strlen(p_weights) == 0 ) 
		w = fvector_unit( 0, 5 );
	else {
		w = fvector_split_3( p_weights, &nc, ",");
		if ( nc != 6 ) {
			printf( "# Error. Weight vector must have 6 elements %i\n", nc );
			exit( 1 );
		}
	}

	/***************************/
       /* Read BLOSUM matrix */   

	blm_qij = read_blosummat_qij( p_blosum_qij, &alphabet );

	k1 = fmatrix_k1( blm_qij );

      /***** READ LIST OF PEPTIDES FROM FILES *********/

      	peplist1 = pairlist2_read( argv[1] );

	if ( ! peplist1 ) {
		printf( "# Error. Cannot read PAIRLIST from %s\n", argv[1] );
		exit( 1 );
	}
 	   	   
       /*encode as blosum indices, and calculate normalization factor for k3*/
       for ( pep1=peplist1; pep1; pep1=pep1->next ) {

	     	pep1->ivcdr1a = ivector( 0, pep1->lcdr1a-1 );

		for ( j=0; j<pep1->lcdr1a; j++) {

			ix = strpos( alphabet, pep1->cdr1a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr1a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr1a[j] = ix; 

	     	}

		pep1->ivcdr2a = ivector( 0, pep1->lcdr2a-1 );

		for ( j=0; j<pep1->lcdr2a; j++) {

			ix = strpos( alphabet, pep1->cdr2a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr2a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr2a[j] = ix; 

	     	}

		pep1->ivcdr3a = ivector( 0, pep1->lcdr3a-1 );

		for ( j=0; j<pep1->lcdr3a; j++) {

			ix = strpos( alphabet, pep1->cdr3a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr3a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr3a[j] = ix; 

	     	}

		pep1->ivcdr1b = ivector( 0, pep1->lcdr1b-1 );

		for ( j=0; j<pep1->lcdr1b; j++) {

			ix = strpos( alphabet, pep1->cdr1b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr1b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr1b[j] = ix; 

	     	}

		pep1->ivcdr2b = ivector( 0, pep1->lcdr2b-1 );

		for ( j=0; j<pep1->lcdr2b; j++) {

			ix = strpos( alphabet, pep1->cdr2b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr2b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr2b[j] = ix; 

	     	}

		pep1->ivcdr3b = ivector( 0, pep1->lcdr3b-1 );

		for ( j=0; j<pep1->lcdr3b; j++) {

			ix = strpos( alphabet, pep1->cdr3b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr3b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr3b[j] = ix; 

	     	}

		pep1->scdr1a = k3_sum(pep1->ivcdr1a, pep1->ivcdr1a, pep1->lcdr1a, pep1->lcdr1a);
		pep1->scdr2a = k3_sum(pep1->ivcdr2a, pep1->ivcdr2a, pep1->lcdr2a, pep1->lcdr2a);
		pep1->scdr3a = k3_sum(pep1->ivcdr3a, pep1->ivcdr3a, pep1->lcdr3a, pep1->lcdr3a);

		pep1->scdr1b = k3_sum(pep1->ivcdr1b, pep1->ivcdr1b, pep1->lcdr1b, pep1->lcdr1b);
		pep1->scdr2b = k3_sum(pep1->ivcdr2b, pep1->ivcdr2b, pep1->lcdr2b, pep1->lcdr2b);
		pep1->scdr3b = k3_sum(pep1->ivcdr3b, pep1->ivcdr3b, pep1->lcdr3b, pep1->lcdr3b);
		
		pep1->score = pep1->scdr1a * w[0] + pep1->scdr2a * w[1] + pep1->scdr3a * w[2] +
				pep1->scdr1b * w[3] + pep1->scdr2b * w[4] + pep1->scdr3b * w[5];
	}

	if ( ! p_all_against_all && strlen( p_db ) > 0 ) {
		peplist2 = pairlist2_read( p_db );

		if ( ! peplist2 ) {
			printf( "# Error. Cannot read Database %s\n", p_db );
			exit( 1 );
		}
	}
	else
		peplist2 = NULL;

	 /*encode as blosum indices, and calculate normalization factor for k3*/
       	for ( pep1=peplist2; pep1; pep1=pep1->next ) {

	     	pep1->ivcdr1a = ivector( 0, pep1->lcdr1a-1 );

		for ( j=0; j<pep1->lcdr1a; j++) {

			ix = strpos( alphabet, pep1->cdr1a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr1a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr1a[j] = ix; 

	     	}

		pep1->ivcdr2a = ivector( 0, pep1->lcdr2a-1 );

		for ( j=0; j<pep1->lcdr2a; j++) {

			ix = strpos( alphabet, pep1->cdr2a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr2a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr2a[j] = ix; 

	     	}

		pep1->ivcdr3a = ivector( 0, pep1->lcdr3a-1 );

		for ( j=0; j<pep1->lcdr3a; j++) {

			ix = strpos( alphabet, pep1->cdr3a[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr3a[j] );
				exit( 1 );
			}

	        	pep1->ivcdr3a[j] = ix; 

	     	}

		pep1->ivcdr1b = ivector( 0, pep1->lcdr1b-1 );

		for ( j=0; j<pep1->lcdr1b; j++) {

			ix = strpos( alphabet, pep1->cdr1b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr1b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr1b[j] = ix; 

	     	}

		pep1->ivcdr2b = ivector( 0, pep1->lcdr2b-1 );

		for ( j=0; j<pep1->lcdr2b; j++) {

			ix = strpos( alphabet, pep1->cdr2b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr2b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr2b[j] = ix; 

	     	}

		pep1->ivcdr3b = ivector( 0, pep1->lcdr3b-1 );

		for ( j=0; j<pep1->lcdr3b; j++) {

			ix = strpos( alphabet, pep1->cdr3b[j] );

			if ( ix < 0 ) {
				printf( "Error. Unknown amino acid %s %c\n", pep1->id, pep1->cdr3b[j] );
				exit( 1 );
			}

	        	pep1->ivcdr3b[j] = ix; 

	     	}

		pep1->scdr1a = k3_sum(pep1->ivcdr1a, pep1->ivcdr1a, pep1->lcdr1a, pep1->lcdr1a);
		pep1->scdr2a = k3_sum(pep1->ivcdr2a, pep1->ivcdr2a, pep1->lcdr2a, pep1->lcdr2a);
		pep1->scdr3a = k3_sum(pep1->ivcdr3a, pep1->ivcdr3a, pep1->lcdr3a, pep1->lcdr3a);

		pep1->scdr1b = k3_sum(pep1->ivcdr1b, pep1->ivcdr1b, pep1->lcdr1b, pep1->lcdr1b);
		pep1->scdr2b = k3_sum(pep1->ivcdr2b, pep1->ivcdr2b, pep1->lcdr2b, pep1->lcdr2b);
		pep1->scdr3b = k3_sum(pep1->ivcdr3b, pep1->ivcdr3b, pep1->lcdr3b, pep1->lcdr3b);
		
		pep1->score = pep1->scdr1a * w[0] + pep1->scdr2a * w[1] + pep1->scdr3a * w[2] +
				pep1->scdr1b * w[3] + pep1->scdr2b * w[4] + pep1->scdr3b * w[5];
	}

       	for (pep1=peplist1; pep1; pep1=pep1->next ) {

		bestp = NULL;
		best_sco = -99.9;

		if ( p_all_against_all )
			peplist_db = pep1;
		else
			peplist_db = peplist2;

		for ( pep2=peplist_db; pep2; pep2=pep2->next ) {

			if ( p_exself && strcmp( pep1->cdr1a, pep2->cdr1a ) == 0 && strcmp( pep1->cdr2a, pep2->cdr2a ) == 0 &&
					strcmp( pep1->cdr3a, pep2->cdr3a ) == 0 && strcmp( pep1->cdr1b, pep2->cdr1b ) == 0 &&
					strcmp( pep1->cdr2b, pep2->cdr2b ) == 0 && strcmp( pep1->cdr3b, pep2->cdr3b ) == 0 )
                                continue;

			if ( p_all_against_all && pep1 == pep2 )
				continue;

			sco_cdr1a = k3_sum(pep1->ivcdr1a, pep2->ivcdr1a, pep1->lcdr1a, pep2->lcdr1a)/sqrt( pep1->scdr1a * pep2->scdr1a );
			sco_cdr2a = k3_sum(pep1->ivcdr2a, pep2->ivcdr2a, pep1->lcdr2a, pep2->lcdr2a)/sqrt( pep1->scdr2a * pep2->scdr2a );
			sco_cdr3a = k3_sum(pep1->ivcdr3a, pep2->ivcdr3a, pep1->lcdr3a, pep2->lcdr3a)/sqrt( pep1->scdr3a * pep2->scdr3a );
			sco_cdr1b = k3_sum(pep1->ivcdr1b, pep2->ivcdr1b, pep1->lcdr1b, pep2->lcdr1b)/sqrt( pep1->scdr1b * pep2->scdr1b );
			sco_cdr2b = k3_sum(pep1->ivcdr2b, pep2->ivcdr2b, pep1->lcdr2b, pep2->lcdr2b)/sqrt( pep1->scdr2b * pep2->scdr2b );
			sco_cdr3b = k3_sum(pep1->ivcdr3b, pep2->ivcdr3b, pep1->lcdr3b, pep2->lcdr3b)/sqrt( pep1->scdr3b * pep2->scdr3b );
			
			
			sco = sco_cdr1a * w[0] + sco_cdr2a * w[1] + sco_cdr3a * w[2] + sco_cdr1b * w[3] +
				sco_cdr2b * w[4] + sco_cdr3b * w[5];

			if ( sco > best_sco ) {
				best_sco = sco;
				bestp = pep2;
			}

			if ( p_all_against_all )
				printf( "ALL Query %s %s %s %s %s %s %s %f Hit %s %s %s %s %s %s %s %f %f\n", 
					pep1->id, pep1->cdr1a, pep1->cdr2a, pep1->cdr3a, pep1->cdr1b, pep1->cdr2b, pep1->cdr3b, 
					pep1->target,
                                      	pep2->id, pep2->cdr1a, pep2->cdr2a, pep2->cdr3a, pep2->cdr1b, pep2->cdr2b, pep2->cdr3b,
                                        sco, pep2->target );
		}

		if ( ! p_all_against_all )
			printf( "Best Query %s %s %s %s %s %s %s %f Hit %s %s %s %s %s %s %s %f %f\n", 
				pep1->id, pep1->cdr1a, pep1->cdr2a, pep1->cdr3a, pep1->cdr1b, pep1->cdr2b, pep1->cdr3b, pep1->target, 
				bestp->id, bestp->cdr1a, bestp->cdr2a, bestp->cdr3a, bestp->cdr1b, bestp->cdr2b, bestp->cdr3b,
				best_sco, bestp->target );
       	}

       	exit( 0 );
}
