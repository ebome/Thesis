#include "bmpio.h"
struct Patch {
  long i; // image containing this patch
  long x, y; // location of patch within image
  long d; // similarity-distance of this patch from the target
  long set(long I, long X, long Y, long D);
};

long Patch::set(long I, long X, long Y, long D) {
  i = I;
  x = X; y = Y;
  return d = D;
}

long patchDifference(const struct BMP *P1, const struct BMP *P2,
long X1, long Y1, long X2, long Y2, long Width, long Height) {
  long x, y;
  long d;
  long g1, g2;
  d = 0;
  for (y = 0; y < Height; y++) {
    for (x = 0; x < Width; x++) {
      g1 = P1->getpixel24g(X1 + x, Y1 + y);
      g2 = P2->getpixel24g(X2 + x, Y2 + y);
      d += (g1 - g2) * (g1 - g2);
    }
  }
return d;
}


int main(void) {
  struct BMP mi, ms, ai, as;
  long **vote;
  long v;
// image sizes
  long XM, YM;
  XM = YM = 256;
// patch sizes, offsets, counts, locality
  long p_X, p_Y, p_dx, p_dy, p_NX, p_NY, p_dxr, p_dyr;
  p_X = p_Y = 16;
  p_dx = p_dy = 8;
  p_NX = ((XM - p_X) / p_dx) + 1;
  p_NY = ((YM - p_Y) / p_dy) + 1;
  p_dxr = (p_X + p_X / 2) / p_dx;
  p_dyr = (p_Y + p_Y / 2) / p_dy;
  
  struct Patch p[40];
  long K_patches;
  long threshold;
  K_patches = 1;
  threshold = K_patches * (p_X / p_dx) * (p_Y / p_dy) / 2 - 0;

  long np, NP;
  long d, dm, pm;

  long x, y;
  long px, py;
  long p2x, p2y;
  
  mi.load("sampleMI.bmp"); // input for manual segmentation
  ms.load("sampleMS.bmp"); // manual segmentation
  ai.load("sampleAI.bmp"); // input for automated segmentation

  // initialise a bitmap to store votes
  vote = long_bmp(XM, YM);
  for (y = 0; y < YM; y++) {
    for (x = 0; x < XM; x++) {
      putpixel(vote, x, y, 0, 0, 0);
    }
  }
  
  for (py = 0; py < p_NY; py++) {
    for (px = 0; px < p_NX; px++) {
    
      NP = 0;

      for (p2y = (py - p_dyr > 0? py - p_dyr: 0);
           p2y <= py + p_dyr && p2y < p_NY;
           p2y++) {
        for (p2x = (px - p_dxr > 0? px - p_dxr: 0);
             p2x <= px + p_dxr && p2x < p_NX;
             p2x++) {
             
          d = patchDifference(&ai, &mi, p_dx * px, p_dy * py, p_dx * p2x,
                               p_dy * p2y, p_X, p_Y);

  // replace worst match, or fill up the array first (if empty)
          if (NP == K_patches) {
            dm = d;
          for (np = 0; np < NP; np++) {
            if (p[np].d > dm) {
               pm = np;
               dm = p[pm].d;
            }
          }
          if (dm > d) {
            p[pm].set(0, p_dx * p2x, p_dy * p2y, d);
          }
        } else {
            p[NP++].set(0, p_dx * p2x, p_dy * p2y, d);
        }
        
      }
    }
    
    for (np = 0; np < NP; np++) {
      for (y = 0; y < p_Y; y++) {
        for (x = 0; x < p_X; x++) {
          v = getpixelG(vote, p_dx * px + x, p_dy * py + y);
          if (ms.getpixel24r(p[np].x + x, p[np].y + y)) {
            v += 1;
          }
          putpixel(vote, p_dx * px + x, p_dx * py + y, v, v, v);
        }
      }
    }
    
  }
}

  // generate segmentation output
  as = ai;
  for (y = 0; y < YM; y++) {
    for (x = 0; x < XM; x++) {
      v = (getpixelG(vote, x, y) > threshold? 255: 0);
      as.putpixel24(x, y, v, v, v);
    }
  }
  as.save("sampleAS.bmp");
  free(vote);

  return 0;
}
