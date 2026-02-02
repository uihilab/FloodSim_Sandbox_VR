// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Engine/GameInstance.h"
#include "Floodsim.generated.h"

/**
 * 
 */
UCLASS()
class VR_VIS_API UFloodsim : public UGameInstance
{
	GENERATED_BODY()
	
public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Files")
	FString JsonString{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool ViewInVR{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool SpawnNPCs{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool SpawnVehicles{};
};
