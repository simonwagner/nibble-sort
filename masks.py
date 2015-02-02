#!/usr/bin/env python

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def mask(count, pos):
    return ((2**(count*4)-1) << (pos*4)) & 0xFFFFFFFFFFFFFFFFL

def main():
    masks = []
    for count in range(17):
        for pos in range(16):
            masks.append(mask(count=count, pos=pos))

    for chunk in chunks(masks, 16):
        print str.join(",", ("0x%016XULL" % mask for mask in chunk)) + ","

if __name__ == "__main__":
    main()
