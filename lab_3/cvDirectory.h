namespace cv
{
	class Directory
	{
	public:
		static std::vector<std::string> GetListFiles(const std::string& path, const std::string & exten = "*", bool addPath = true);
		static std::vector<std::string> GetListFilesR(const std::string& path, const std::string & exten = "*", bool addPath = true);
		static std::vector<std::string> GetListFolders(const std::string& path, const std::string & exten = "*", bool addPath = true);
	};
}