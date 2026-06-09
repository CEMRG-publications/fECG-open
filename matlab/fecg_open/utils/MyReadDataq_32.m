function a = MyReadDataq_32(filestr)
% MyReadDataq_32  Read a little-endian int32 binary channel file.
%
% PURPOSE
%   Reads one Monica DK .ch* binary file (little-endian, signed 32-bit
%   integers) and returns the samples as a double-precision column vector.
%   Prints a diagnostic message and returns without error if the file is
%   not found.
%
% INPUTS
%   filestr - (string) full path to the binary .ch* file.
%
% OUTPUTS
%   a       - (M x 1 double) signal samples converted from int32.
%             Returns uninitialized / empty if the file cannot be opened.

    fi = fopen(filestr, 'r', 'l');
    if fi == -1
        disp([filestr, ' file not found!']);
        return;
    end
    a = fread(fi, 'int32');
    fclose(fi);
end
