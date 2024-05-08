import torch


def main():
    # load the library, assume it is located together with this file
    torch.ops.load_library("../libCANDY.so")
    # gen the input tensor
    torch.ops.CANDY.index_create("idx1", "flat")
    torch.ops.CANDY.index_ediCfgI64("idx1", "vecDim", 4)
    a = torch.rand(1, 4)
    b = torch.rand(1, 4)
    torch.ops.CANDY.index_init("idx1")
    torch.ops.CANDY.index_insert("idx1", a)
    torch.ops.CANDY.index_insert("idx1", b)
    c = torch.ops.CANDY.index_search("idx1", a, 1)
    print("rawData", torch.ops.CANDY.index_rawData("idx1"))
    print("search result", c)
    torch.ops.CANDY.tensorToFile(c[0], "c.rbt")
    print("loaded result")
    d = torch.ops.CANDY.tensorFromFile("c.rbt")
    print(d)


if __name__ == "__main__":
    main()
