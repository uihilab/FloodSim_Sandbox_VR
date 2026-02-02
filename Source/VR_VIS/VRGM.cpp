// Fill out your copyright notice in the Description page of Project Settings.


#include "VRGM.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

void AVRGM::Deserealize()
{
	if (FFileHelper::LoadFileToString(JsonString, *FloodmapPath))
	{
		UE_LOG(LogTemp, Display, TEXT("Successfully loaded Json from path to string."));
		UE_LOG(LogTemp, Display, TEXT("%s"), *JsonString);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load Json from path to string."));
	}
}