# Prompt tests

One task per row; write the outcome of each variant in its cell (solved / near / wrong, and what the answer got wrong).

| task | grade | p0 | p1 | p3 | p4 | p5 | p6 | (perm: same columns) |
|---|---|---|---|---|---|---|---|---|
| 009d5c81 | easy |  |  |  |  |  |  | |
| 00dbd492 | easy |  |  |  |  |  |  | |
| 0a2355a6 | medium |  |  |  |  |  |  | |
| 37d3e8b2 | easy |  |  |  |  |  |  | |
| 54db823b | easy |  |  |  |  |  |  | |
| 60a26a3e | easy |  |  |  |  |  |  | |
| 64a7c07e | easy |  |  |  |  |  |  | |
| 6ea4a07e | easy |  |  |  |  |  |  | |
| 84f2aca1 | easy |  |  |  |  |  |  | |
| ae58858e | easy |  |  |  |  |  |  | |

## Rules

**009d5c81** - rule: The small object of colour 1 is a key: its cells become 0. The larger object of colour 8 is recoloured according to the key's shape - the plus shape gives 2, the shape 101/010/111 gives 3, the shape 111/101/010 gives 7.

wrong rule (P5): The object of colour 8 becomes 2 and the object of colour 1 is removed.

**00dbd492** - rule: Each square ring of colour 2 has an empty interior, which is filled by the ring's size: a 3x3 interior gets 8, a 5x5 interior gets 4, a 7x7 interior gets 3. The single centre cell of colour 2 stays.

wrong rule (P5): The interior of every square ring of colour 2 is filled with 8.

**0a2355a6** - rule: Every shape of colour 8 is recoloured by how many holes (enclosed empty regions) it has: 1 hole gives 1, 2 holes give 3, 3 holes give 2, 4 holes give 4.

wrong rule (P5): A shape of colour 8 that has any hole becomes 1; a shape without holes stays 8.

**37d3e8b2** - rule: Every shape of colour 8 is recoloured by how many holes (enclosed empty regions) it has: 1 hole gives 1, 2 holes give 2, 3 holes give 3, 4 holes give 7.

wrong rule (P5): Every shape of colour 8 is recoloured to the number of its holes: 1 hole gives 1, 2 holes give 2, 3 holes give 3, 4 holes give 4.

**54db823b** - rule: The grid holds several separate patterns of colours 3 and 9. The pattern with the fewest 9 cells is erased (all its cells become 0); the others stay.

wrong rule (P5): The pattern with the fewest cells of any colour is erased (all its cells become 0); the others stay.

**60a26a3e** - rule: The 2 shapes are small diamonds. Where two neighbouring diamonds share a row or a column, with no diamond between them, the empty cells between their nearest tips are filled with 1.

wrong rule (P5): The empty cells between every two diamonds that share a row or a column are filled with 1, including two diamonds that have another diamond between them.

**64a7c07e** - rule: Every object (a group of same-coloured cells touching sideways or diagonally) moves to the right by its own width in cells, keeping its shape and colour.

wrong rule (P5): Every object moves one cell to the right, keeping its shape and colour.

**6ea4a07e** - rule: The input holds one non-zero colour. Every cell of that colour becomes 0, and every cell that was 0 becomes a colour that depends on the input's colour: 8 gives 2, 3 gives 1, 5 gives 4.

wrong rule (P5): The two colours swap: every 0 becomes the input's non-zero colour and every cell of that colour becomes 0.

**84f2aca1** - rule: Each closed ring has a hole of empty (0) cells inside. A hole of one cell is filled with 5; a hole of two cells is filled with 7.

wrong rule (P5): Every hole inside a closed ring is filled with 5.

**ae58858e** - rule: Every connected group of 2 cells (cells touching sideways or diagonally) that has 4 or more cells becomes 6. Smaller groups stay 2.

wrong rule (P5): The largest group of 2 cells becomes 6; every other group stays 2.
