# nonce-cuda

Represents a Cuda C application that computes for a given prefix a suffix that respects the following condition: the sha1 function applied to the entire string (prefix + suffix) produces a string (sha1 digest) that ends with some specific characters

The sha1 values are computed on the GPU and the prefixes are generated on the CPU (via a BFS algorithm)
