# OurLattice Package

This is a package containing Pandas accessors to help dealing with lattice objects such as
polytopes and faces. 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
```bash
pip install ourlattice
```

## Usage
The `ourlattice` Python package is a [Pandas](https://pandas.pydata.org/) extension that has lot of features for [polytopes](https://en.wikipedia.org/wiki/Polytope). Start by define your polytope.

```python
    from ourlattice.accessors.lattice import SupportFieldType
    from ourlattice.accessors.polytope import Polytope

    df = Polytope.construct(
        id="my-polytope",
        constraints=[
            {
                "a": 1,
                SupportFieldType.B.value: 1,
                SupportFieldType.ID.value: 0,
            },
            {
                "a": -1,
                "b":  1,
                "c":  1,
                SupportFieldType.B.value: 0,
                SupportFieldType.ID.value: 1,
            },
            {
                "x":  -1,
                "y":  -1,
                "z:": -1,
                SupportFieldType.B.value: -1,
                SupportFieldType.ID.value: 2,
            },
            {
                "x":  1,
                "y":  1,
                "z:": 1,
                SupportFieldType.B.value: 1,
                SupportFieldType.ID.value: 2,
            },
        ]
    )
```
```python
>>> df
#            a    #b  b    c    x    y    z:
#id #r  #w                                   
0   NaN NaN  1.0   1  0.0  0.0  0.0  0.0  0.0
1   NaN NaN -1.0   0  1.0  1.0  0.0  0.0  0.0
2   NaN NaN  0.0  -1  0.0  0.0 -1.0 -1.0 -1.0
        NaN  0.0   1  0.0  0.0  1.0  1.0  1.0
```
Then run a simple check if a point is in the polytope:
```python
>>> df.polytope.isin({"a": 1})
False

>>> df.polytope.isin({"a": 1, "b": 1, "c": 1, "x": 1})
True
```

Every polytope is indexed by an [required] `#id`, description tag `#r` and a weight `#w`. Each constraint must have the support vector `#b` and at least one variable constant. After the polytope is defined, you can compute various functions. For example, `assume` takes a list of variables and assumes that they are set:

```python
>>> df.polytope.assume(["a"])
#            #b    b    c    x    y   z:
#id #r  #w                               
0   NaN NaN  0.0  0.0  0.0  0.0  0.0  0.0
1   NaN NaN  1.0  1.0  1.0  0.0  0.0  0.0
2   NaN NaN -1.0  0.0  0.0 -1.0 -1.0 -1.0
        NaN  1.0  0.0  0.0  1.0  1.0  1.0
```

Another example is `tautologies` which finds the constraints in the polytope that are true no matter the interpretation:
```python
>>> df.polytope.assume(["a", "b", "c"]).polytope.tautologies()
#              #b    x    y   z:
#id #r  #w                     
0   NaN NaN  0.0  0.0  0.0  0.0
1   NaN NaN -1.0  0.0  0.0  0.0
2   NaN NaN  1.0  1.0  1.0  1.0
```

## Contributing and todo's
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)