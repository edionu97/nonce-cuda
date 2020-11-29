#pragma once
namespace sha_macros
{
	#define SHA1_ROL(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

	#define SHA1_BLK0(i) (block->l[i] = (SHA1_ROL(block->l[i],24)&0xFF00FF00) | (SHA1_ROL(block->l[i],8)&0x00FF00FF))
	#define SHA1_BLK(i) (block->l[(i)&15] = SHA1_ROL(block->l[(i+13)&15]^block->l[(i+8)&15] ^block->l[(i+2)&15] ^ block->l[i&15],1))

	#define SHA1_R0(v,w,x,y,z,i) z+= (((w)&((x)^(y)))^(y)) + SHA1_BLK0(i) + 0x5A827999 + SHA1_ROL(v,5); (w) = SHA1_ROL(w,30)
	#define SHA1_R1(v,w,x,y,z,i) z+=(((w)&((x)^(y)))^(y)) + SHA1_BLK(i)  + 0x5A827999+ SHA1_ROL(v,5); (w) = SHA1_ROL(w,30)
	#define SHA1_R2(v,w,x,y,z,i) z+=((w)^(x)^(y)) + SHA1_BLK(i) +  0x6ED9EBA1 + SHA1_ROL(v,5); (w) = SHA1_ROL(w,30)
	#define SHA1_R3(v,w,x,y,z,i) z+=((((w)|(x))&(y))|((w)&(x))) + SHA1_BLK(i) + 0x8F1BBCDC + SHA1_ROL(v,5);(w) = SHA1_ROL(w,30)
	#define SHA1_R4(v,w,x,y,z,i) z+=((w)^(x)^(y)) + SHA1_BLK(i) + 0xCA62C1D6 + SHA1_ROL(v,5); (w) = SHA1_ROL(w,30)
}