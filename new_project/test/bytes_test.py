string = '我是谁'
string1 = '啊'

bytes_a = string.encode('gb2312')
bytes_1 = string1.encode('gb2312')
print(bytes_1)
int_byte_1 = int.from_bytes(bytes_1,byteorder='little')
print("int_byte:",int_byte_1)
int_list = []
for i in range(3):
    bytes_b = bytes_a[2*i:2*(i+1)]
    int_byte = int.from_bytes(bytes_b,byteorder='little')
    int_list.append(int_byte)
print(int_list)
byte_from_int = int_byte_1.to_bytes(2,byteorder='little')
print("byte_from_int:",byte_from_int)