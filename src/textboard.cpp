
#include <fstream>
#include <algorithm>


template<typename obj>
void writeFile(std::ofstream& out, obj& t)
{
  out.write(reinterpret_cast<char*>(&t), sizeof(obj));
}


std::ofstream file("objData.dat");

 writeFile<long>(file, obj.ID.size());
 std::for_each(obj.ID.begin(), obj.ID.end(), std::bind1st(writeFile<s>, file));
 file.close();