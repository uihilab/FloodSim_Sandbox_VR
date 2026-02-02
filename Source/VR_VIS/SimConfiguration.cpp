// Fill out your copyright notice in the Description page of Project Settings.


#include "SimConfiguration.h"
#include "HAL/FileManager.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Floodsim.h"

// Sets default values
ASimConfiguration::ASimConfiguration()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;

}

// Called when the game starts or when spawned
void ASimConfiguration::BeginPlay()
{
	Super::BeginPlay();
	ReadFiles();
}

void ASimConfiguration::ReadFiles()
{
	FileNames.Empty();
	// FString Directory = FPaths::ProjectDir() / TEXT("Content");
	FString Directory = FPaths::LaunchDir() / TEXT("Floodmaps");
	IFileManager& FileManager = IFileManager::Get();
	FString FileExtension = TEXT(".geojson");

	if (FileManager.DirectoryExists(*Directory))
	{
		UE_LOG(LogTemp, Display, TEXT("Successfully read directory."));

		// FilePaths will only contain the string of the file name. Must also get full directory path for conversion of json to string in VRGM

		FileManager.FindFiles(FileNames, *Directory, *FileExtension);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Specified directory not found"));
	}
}

FString ASimConfiguration::GetFullPath() const
{
	FString FilePath = FPaths::LaunchDir() / TEXT("Floodmaps") / (TEXT("%s"), *SelectedName);

	// FString FilePath = 

	UE_LOG(LogTemp, Display, TEXT("%s"), *SelectedName);
	UE_LOG(LogTemp, Display, TEXT("%s"), *FilePath);

	return FilePath;
}

void ASimConfiguration::SubmitConfig()
{
	FString path = SelectedName;
	FString FileContent{};
	UFloodsim* FloodInstance = Cast<UFloodsim>(GetGameInstance());
	FFileHelper::LoadFileToString(FileContent, *path);

	if (FloodInstance)
	{
		FloodInstance->JsonString = FileContent;
		FloodInstance->ViewInVR = UsingVR;
		FloodInstance->SpawnNPCs = NPCConfig;
		FloodInstance->SpawnVehicles = VehicleConfig;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Could not cast game instance to Floodsim subclass."));
	}
}

bool ASimConfiguration::SelectFiles(FString& OutFileName)
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();

	if (DesktopPlatform)
	{
		// Set up dialogue options
		FString DefaultPath = FPaths::ProjectDir();
		FString FileTypes = TEXT("GeoJSON files (.geojson)|*.geojson");
		TArray<FString> OutFileNames{};

		// Open file dialogue

		bool Opened = DesktopPlatform->OpenFileDialog(nullptr, TEXT("Select a File"), DefaultPath, TEXT(""), FileTypes, EFileDialogFlags::None, OutFileNames);

		if (Opened && OutFileNames.Num() > 0)
		{
			OutFileName = OutFileNames[0];
			UE_LOG(LogTemp, Display, TEXT("File chosen: %s"), *OutFileName);
			ShortName = FPaths::GetBaseFilename(OutFileName);
			return true;
		}
	}

	UE_LOG(LogTemp, Warning, TEXT("No file selected."));
	return false;
}

// Called every frame
void ASimConfiguration::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

