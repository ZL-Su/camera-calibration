#pragma once
#include <string>
#include <vector>
#include <memory>
#include <initializer_list>
#include "properties.hpp"

namespace zl { 
namespace types {
	using vstr_t  = std::vector<std::string>; 
	using vvstr_t = std::vector<vstr_t>;
	template<typename _Ty>
	using initlist = std::initializer_list<_Ty>;
}
namespace io {

	template<typename _FnameContainer> class Dir_
	{
		using My_t = Dir_;
	public:
		using names_t = _FnameContainer;
		inline Dir_() = default;
		inline Dir_(const std::string& _fpath, const names_t& _fnames)
			:m_fpath(_fpath), m_fnames(new names_t(_fnames)) 
		{
			m_type = dgelom::type_name<_FnameContainer>();
		}
		//_list = {{path}, {fnames of cam-1}, {fnames of cam-2}, ...}
		inline Dir_(const types::initlist<types::vstr_t> _list)
			:m_fpath(_list.begin()->front()), m_type(dgelom::type_name<_FnameContainer>())
		{
			names_t vvnams(_list.size() - 1);
			std::size_t i = 1;
			for (auto& vnams : vvnams)
			{
				vnams = std::move(*(_list.begin() + i++));
			}
			m_fnames = std::make_shared<names_t>(vvnams);
		}
		inline Dir_(My_t&& _other)
			: m_fpath(_other.m_fpath), m_fnames(std::move(_other.m_fnames)) 
		{
			m_type = dgelom::type_name<_FnameContainer>();
		}
		//_list = {{path}, {fnames of cam-1}, {fnames of cam-2}, ...}
		inline Dir_& operator= (const types::initlist<types::vstr_t> _list)
		{
			m_fpath = std::move(_list.begin()->front());
			names_t vvnams(_list.size() - 1);
			std::size_t i = 1;
			for (auto& vnams : vvnams)
			{
				vnams = std::move(*(_list.begin() + i++));
			}
			m_fnames = std::make_shared<names_t>(vvnams);

			return (*this);
		}

		__property__(std::string&, fpath);
		__set__(fpath) { m_fpath = fpath; }
		__get__(fpath) { return (m_fpath); }
		__property__(std::shared_ptr<names_t>, fnames);
		__set__(fnames) { m_fnames = fnames; }
		__get__(fnames) const { return (m_fnames); }

	public:
		std::string m_fpath;   //: file absolute path, end with "/"
		std::shared_ptr<names_t> m_fnames; //: generic file names' container
		std::string m_type; //: container type
	};
}}
