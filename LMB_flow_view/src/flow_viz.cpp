/**
  * The following routines are taken from Sintel and Middlebury benchmark
  * for optical flow visualization. Flow is usually vizualized by placing
  * the flow vector in a color circle. Color then indicates direction and
  * brightness indicates magnitude.
  *
  * Middlebury Vizualization:
  *   Best used for computer screens, the center of the color circle is
  *   black (i.e. black indicates zero motion)
  *
  * Sintel Vizualization:
  *   Best used for printing, the center of the color circle is
  *   white (i.e. white indicates zero motion)
  *
  * The following routines perform the conversion:
  *
  *   void flow_viz_middlebury(const float* input,
  *                            unsigned char* output,
  *                            int width,
  *                            int height,
  *                            float scale)
  *
  *   void flow_viz_sintel(const float* input,
  *                            unsigned char* output,
  *                            int width,
  *                            int height,
  *                            float scale)
  *
  * Input is the flow field, with dim order C, W, H.
  * Output is the RGB imag with dim order C, W, H.
  *
  * The scale parameter is needed to scale the flows of your application
  * for apropriate visualization. Its value corresponds to the flow magnitude
  * at maximum saturation (choose aprpopriately for small or large flows).
  *
  */

/// Code from sintel bechmark
/// ---------------------------------------------------

static int ncols = 0;
#define MAXCOLS 60
static int colorwheel[MAXCOLS][3];

static void setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

static void makecolorwheel(void)
{
    /// relative lengths of color transitions:
    /// these are chosen based on perceptual similarity
    /// (e.g. one can distinguish more shades between red and yellow
    ///  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,          255*i/RY,     0,            k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,          0,            k++);
    for (i = 0; i < GC; i++) setcols(0,            255,          255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,            255-255*i/CB, 255,          k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,     0,            255,          k++);
    for (i = 0; i < MR; i++) setcols(255,          0,            255-255*i/MR, k++);
}

static void sintel_cartesian_to_RGB(float fx, float fy, float* pix)
{
    if (ncols == 0) makecolorwheel();

    /// Adjust flow strength to avoid early color saturation
    fx /= 100.f;
    fy /= 100.f;

    if (isnan(fx) || isnan(fy)) {
        fx = 0.f;
        fy = 0.f;
    }

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    if(isnan(fk))
    {
        printf("Bad value (%f,%f)\n",fx,fy);
        return;
    }
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    int b;
    for (b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0;
        float col1 = colorwheel[k1][b] / 255.0;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); /// increase saturation with radius
        pix[b] = col;
    }
}

/// ---------------------------------------------------
/// End of code from sintel bechmark


void flow_viz_sintel(const float* input, unsigned char* output, int width, int height, float scale)
{
    int size=width*height;
    int i;

    float pix[3] = {0.f, 0.f, 0.f};

    scale = 100.0/scale;

    for (i = 0; i < size; ++i) {
        const float* inpixel = &(input[i*2]);
        unsigned char* outpixel = &(output[i*3]);

        sintel_cartesian_to_RGB(inpixel[0]*scale, inpixel[1]*scale, pix);

        outpixel[0] = (unsigned char)(pix[0]*255.f);
        outpixel[1] = (unsigned char)(pix[1]*255.f);
        outpixel[2] = (unsigned char)(pix[2]*255.f);
    }
}


/// Code from middlebury bechmark
/// ---------------------------------------------------

static void middlebury_cartesian_to_RGB(float x, float y, float* R, float* G, float* B)
{
    if (isnan(x) || isnan(y)) {
        x = 0.f;
        y = 0.f;
    }
    const float Pi = 3.1415926536f;
    float radius = sqrt (x * x + y * y);
    if (radius > 1) radius = 1;
    float phi;
    if (x == 0.0)
        if (y >= 0.0) phi = 0.5 * Pi;
        else phi = 1.5 * Pi;
    else if (x > 0.0)
        if (y >= 0.0) phi = atan (y/x);
        else phi = 2.0 * Pi + atan (y/x);
    else phi = Pi + atan (y/x);
    float alpha, beta;    // weights for linear interpolation
    phi *= 0.5;
    // interpolation between red (0) and blue (0.25 * Pi)
    if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
        beta  = phi / (0.125 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha * 255.0 + beta * 255.0));
        (*G) = (int)(radius * (alpha *   0.0 + beta *   0.0));
        (*B) = (int)(radius * (alpha *   0.0 + beta * 255.0));
    }
    if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
        beta  = (phi-0.125 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha * 255.0 + beta *  64.0));
        (*G) = (int)(radius * (alpha *   0.0 + beta *  64.0));
        (*B) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    }
    // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
    if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
        beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha *  64.0 + beta *   0.0));
        (*G) = (int)(radius * (alpha *  64.0 + beta * 255.0));
        (*B) = (int)(radius * (alpha * 255.0 + beta * 255.0));
    }
    if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
        beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha *   0.0 + beta *   0.0));
        (*G) = (int)(radius * (alpha * 255.0 + beta * 255.0));
        (*B) = (int)(radius * (alpha * 255.0 + beta *   0.0));
    }
    // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
    if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
        beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha * 0.0   + beta * 255.0));
        (*G) = (int)(radius * (alpha * 255.0 + beta * 255.0));
        (*B) = (int)(radius * (alpha * 0.0   + beta * 0.0));
    }
    // interpolation between yellow (0.75 * Pi) and red (Pi)
    if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
        beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        (*R) = (int)(radius * (alpha * 255.0 + beta * 255.0));
        (*G) = (int)(radius * (alpha * 255.0 + beta *   0.0));
        (*B) = (int)(radius * (alpha * 0.0   + beta *   0.0));
    }
    if ((*R)<0) R=0;
    if ((*G)<0) G=0;
    if ((*B)<0) B=0;
    if ((*R)>255) (*R)=255;
    if ((*G)>255) (*G)=255;
    if ((*B)>255) (*B)=255;
}

/// ---------------------------------------------------
/// End of code from middlebury bechmark

void flow_viz_middlebury(const float* input, unsigned char* output, int width, int height, float scale)
{
    int size=width*height;
    int i;
    float pix[3] = {0.f, 0.f, 0.f};

    // Eddy: Scale parameter corresponds to maximum possible displacement to display
    float mid_scale = 1.0/scale;

    for (i = 0; i < size; ++i) {
        const float* inpixel = &(input[i*2]);
        unsigned char* outpixel = &(output[i*3]);

        middlebury_cartesian_to_RGB(inpixel[0]*mid_scale, inpixel[1]*mid_scale,
                &pix[0], &pix[1], &pix[2]);

        outpixel[0] = (unsigned char)(pix[0]);
        outpixel[1] = (unsigned char)(pix[1]);
        outpixel[2] = (unsigned char)(pix[2]);
    }
}

