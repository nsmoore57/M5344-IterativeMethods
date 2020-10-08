
# --------------------------------------------------------------------------
# A simple class for conforming triangular meshes. The class is
# designed for simplicity of constructing the mesh.
#
# TODO: I plan to write another mesh class with compressed storage. The user
# would then use LoadableMesh() to build the mesh and then write it into a
# compressed mesh format.
#
# Katharine Long, Sep 2020
# For Math 5344
# --------------------------------------------------------------------------
class LoadableMesh:

    # Initialize an empty mesh.
    def __init__(self):
        # We store vertices as list of (x,y) coordinate pairs. The pairs are
        # stored as tuples (x,y) rather than lists [x,y] so that they can
        # be used as hashable keys.
        # Example: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        # corners of the unit square
        self.verts = []

        # Store elements as list of (a,b,c) vertex index triplets, ordered
        # counterclockwise in the triangle. The triplets are
        # stored as tuples (a,b,c) rather than lists [a,b,c] so that they can
        # be used as hashable keys.
        # Example: [(0, 1, 2), (2, 3, 0)] for the triangulation of the unit
        # square with diagonal running from (0.0,0.0) to (1.0, 1.0)
        self.elems = []

        # Store sides as list of (p,q) vertex index pairs, in sorted order.
        # The pairs are stored as tuples (x,y) rather than lists [x,y] so
        # that they can be used as hashable keys.
        # Example: [(0,1), (1,2), (2,3), (0,3), (0,2)] for the edges in the
        # triangulation of the unit square
        self.sides = []

        # Side sets are stored in a dictionary with the label as key
        # and the set of side indices having that label as the value
        # Example: {0 : set([4]),1 : set([0,2]), 2 : set([1,3])}
        self.sideSets = {}

        # Dictionary that maps vertex's position (x,y) to vertex index
        # Example: {(0.0, 0.0) : 0, (1.0, 0.0) : 1, (1.0, 1.0) : 2, (0.0, 1.0) : 3}
        self.vertToIndexMap = {}

        # Dictionary that maps side's vertex pair (p,q) to side index
        # Example: {(0,1) : 0, (1,2) : 1, (2,3) : 2, (0, 3) : 3, (0,2) : 4}
        self.sideToIndexMap = {}

        # For each vertex, list the elements that are attached (cofacets)
        # Example: [set([0,1]), set([0]), set([0,1]), set([1])]
        self.connectedElemsForVert = []

        # For each side, list the elements that are attached.
        # Example: [set([0]), set([1]), set([1]), set([0,1])]
        self.connectedElemsForSide = []


    # Add a new vertex to the mesh. Vertex is input as (x,y) or [x,y]
    def addVertex(self, vert):
        # Copy the pair into a tuple (just in case it's not already a tuple)
        v = tuple(vert)
        # Ensure that the vertex isn't a duplicate
        if v in self.vertToIndexMap:
            raise RuntimeError('Added vertex (%g,%g) twice' % v)
        else:
            # Assign an index to the new vertex
            vertIndex = len(self.verts)
            # Store the mapping (x,y) <==> index
            self.vertToIndexMap[v] = vertIndex
            self.verts.append(v)
            # Allocate an empty set for the set of connected elements
            self.connectedElemsForVert.append(set())

        # Return the index assigned to this vertex
        return vertIndex


    # Add a new side
    def addSide(self, a, b, label):
        # The side will be identified by its sorted vertices
        side = [a,b]
        side.sort()
        s = tuple(side)
        # Get an index for the new side
        index = len(self.sides)
        # Set up the mappings (p,q) <==> index
        self.sides.append(s)
        self.sideToIndexMap[s] = index

        # Allocate an empty set for the set of connected elements
        self.connectedElemsForSide.append(set())

        # Put this side in the set of sides associated with its label.
        # If that set doesn't exist yet, create it
        if not label in self.sideSets:
            self.sideSets[label] = set([index])
        else:
            self.sideSets[label].add(index)

        # Return the index assigned to this side
        return index


    # Add a new element to the mesh
    def addElem(self, a, b, c):

        # Get the index for the new element
        elemIndex = len(self.elems)

        # Put the indices into a tuple
        abc = (a,b,c)

        # Store the new element
        self.elems.append(abc)

        # For each of the vertices, add this element to the vertex's
        # set of connected elements
        for v in abc:
            self.connectedElemsForVert[v].add(elemIndex)

        # Add each of the element's sides to the list of sides, and then
        # record that this element is attached to that side
        for s in ( [a,b], [b,c], [c,a] ):
            # Sort the side's vertices
            s.sort()
            # Sanity check: side should have already been added
            sKey = tuple(s)
            if sKey not in self.sideToIndexMap:
                raise RuntimeError('side (%d,%d) not in mesh' % sKey)
            # Add this element to the side's set of connected elements
            sideIndex = self.sideToIndexMap[sKey]
            self.connectedElemsForSide[sideIndex].add(elemIndex)

        # Return the index assigned to this element
        return elemIndex


    # Look up the label for a side
    def getSideLabel(self, side):
        sideIndex = self.sideToIndexMap[side]
        for label, ss in self.sideSets.items():
            if sideIndex in ss:
                return label
        return 0


    # Dump the internal data
    def dump(self):

        print('Vertices: num=%d' % len(self.verts))
        for v,cf in zip(self.verts, self.connectedElemsForVert):
            print('\t', v, ' cofacets=', cf)

        print('Elements: num=%d' % len(self.elems))
        for e in self.elems:
            print('\t', e)

        print('Sides: num=%d' % len(self.sides))
        for s, cf in zip(self.sides, self.connectedElemsForSide):
            print('\t', s, ' cofacets=', cf)

        print('Side Sets: num=%d' % len(self.sideSets))
        for label, sides in self.sideSets.items():
            print('\tlabel=', label, ' sides', sides)

# ------------------------------------------------------------------------
# Create a simple two-element square for use in testing
#
#     3 ---- 2
#     |    / |
#     |   /  |
#     |  /   |
#     | /    |
#     0 ---- 1
#
# Vertices:
# 0 (0,0)
# 1 (1,0)
# 2 (1,1)
# 3 (0,1)
#
# Sides: (sorted vertices)
# 0 (0,1)
# 1 (1,2)
# 2 (2,3)
# 3 (0,3)
# 4 (0,2)
#
# Side sets:
# 0 {4}
# 1 {0,2}
# 2 {1,3}
#
# Elements: (vetices in CCW order)
#
# 0 (0,1,2)
# 1 (0,2,3)
#
#
def TwoElemSquare():

    # Create empty mesh
    mesh = LoadableMesh()

    # Add vertices, giving (x,y) pairs
    mesh.addVertex((0,0))
    mesh.addVertex((1,0))
    mesh.addVertex((1,1))
    mesh.addVertex((0,1))

    # Add sides v1, v2, label (sorting not needed at this point)
    mesh.addSide(0, 1, 1) # south
    mesh.addSide(1, 2, 2) # east
    mesh.addSide(2, 3, 1) # north
    mesh.addSide(3, 0, 2) # west
    mesh.addSide(0, 2, 0) # (0,0) to (1,1)

    # Add the elements
    mesh.addElem(0,1,2)
    mesh.addElem(0,2,3)

    # Return filled-in mesh
    return mesh

# ---------------------------------------------------------------------------
# Test code

if __name__=='__main__':

    mesh = TwoElemSquare()
    mesh.dump()
