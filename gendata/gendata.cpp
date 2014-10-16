#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <time.h>

using namespace std;
int main(void)
{
    int m = 10000, n = 10000;
    FILE *fp = fopen("data.mat", "w");
    srand((unsigned int)time(0));
    if (fp != NULL) {
        fprintf(fp, "%d %d\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float temp = float(rand()) / RAND_MAX * 0.2 - 0.1;
                fprintf(fp, "%f ", temp);
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}
