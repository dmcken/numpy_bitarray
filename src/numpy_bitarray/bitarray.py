'''Bit array in numpy implementation'''

# System imports
import math

# External imports
import numpy


class Bitarray:
    """Bit array using numpy as backend."""

    _chunk_type = numpy.uint8
    _chunk_size = 8

    def __init__(self, size: int = 0) -> None:
        """Constructor.

        Args:
            size (int, optional): Size of bitarray. Defaults to 0.
        """
        self._size = size
        chunk_count = math.ceil(size / float(self._chunk_size))
        self._data = numpy.zeros(
            chunk_count,
            dtype=self._chunk_type,
        )

    def _split_index(self, index: int) -> (int,int):
        """Split single index to components with error checking.

        Args:
            index (int): _description_
            int (_type_): _description_
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Invalid index {index} is out-of-bounds")
        chunk_index = math.floor(index / self._chunk_size)
        chunk_subindex = index % self._chunk_size

        return (chunk_index, chunk_subindex)

    def __getitem__(self, index: int) -> bool:
        """Get value from specificed location.

        Args:
            index (int): Location to fetch.

        Returns:
            bool: value at location.
        """
        chunk_index, chunk_subindex = self._split_index(index)
        chunk_bits = numpy.unpackbits(
            self._data[chunk_index],
            bitorder='little',
        )
        return chunk_bits[chunk_subindex]

    def __setitem__(self, index: int, value: bool) -> None:
        """Set value at specified location.

        Args:
            index (int): Location to set.
            value (bool): Value to set.
        """
        chunk_index, chunk_subindex = self._split_index(index)

        chunk_bits = numpy.unpackbits(
            self._data[chunk_index],
            bitorder='little',
        )

        # Standardize all truthy values to 0 and 1
        if value:
            chunk_bits[chunk_subindex] = 1
        else:
            chunk_bits[chunk_subindex] = 0

        self._data[chunk_index] = numpy.packbits(
            chunk_bits,
            bitorder='little',
        )[0]

    def save(self, filename: str, compressed: bool = True) -> None:
        """Save bitset to file.

        Args:
            filename (str): Filename to save bitarray to.
            compressed (bool, optional): If to compress file. Defaults to True.
        """
        with open(filename, mode='wb') as f:
            if compressed:
                numpy.savez(f, self._data)
            else:
                numpy.save(f, self._data)

    def load(self, filename: str):
        """Load bitset into file.

        This overwrites the current state of the bitset.

        Args:
            filename (str): _description_
        """
        with open(filename, 'rb') as f:
            numpy.load(f, self._data)

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: output.
        """
        return str(numpy.unpackbits(self._data))

    def get_readonly(self) -> bool:
        """Get read-only status.

        Returns:
            bool: _description_
        """
        return self._data.flags.writeable

    def set_readonly(self, value: bool) -> None:
        """Set read-only status.

        Args:
            value (bool): Read-only status to set.
        """
        self._data.flags.writeable = value

    readonly = property(get_readonly, set_readonly)

    def bitcount(self) -> int:
        """Count the number of non-zero entries.

        Returns:
            int: final count.
        """
        return numpy.sum(numpy.bitwise_count(self._data))
