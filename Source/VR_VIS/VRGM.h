// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "VRGM.generated.h"

/**
 * 
 */
UCLASS()
class VR_VIS_API AVRGM : public AGameModeBase
{
	GENERATED_BODY()
	
public:

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Files")
	FString FloodmapPath{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Files")
	FString JsonString{};

	UFUNCTION(BlueprintCallable, Category = "Json")
	void Deserealize();
};
