//
// Created by huangshize on 16-4-25.
//

#ifndef JAFFE_RNG_H
#define JAFFE_RNG_H


#include <algorithm>
#include <iterator>

#include "common.h"

namespace jaffe {

    typedef boost::mt19937 rng_t;

    inline rng_t *jaffe_rng() {
        return static_cast<jaffe::rng_t *>(Jaffe::rng_stream().generator());
    }

}
#endif //JAFFE_RNG_H
