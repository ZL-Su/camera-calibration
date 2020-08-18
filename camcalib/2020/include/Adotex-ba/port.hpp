/********************************************************************
  Dgelom Adotex-ba, a fast bundle adjustment enhanced by Adotex
  Copyright(C) 2020, Dgelom Su, all rights reserved.
 *******************************************************************/
#pragma once
#include<string>

#ifdef DGELOM_ADOTEX_BA_EXPORTS
#define ADOTEX_BA_API __declspec(dllexport)
#else
#define ADOTEX_BA_API __declspec(dllimport)
#endif

#define DGELOM_NAMESPACE_BEGIN namespace dgelom{
#define DGELOM_NAMESPACE_END }

DGELOM_NAMESPACE_BEGIN
struct license_type {
	std::string type;
	std::string isn;
	bool activated = false;
};
ADOTEX_BA_API license_type _Get_license() noexcept;
DGELOM_NAMESPACE_END