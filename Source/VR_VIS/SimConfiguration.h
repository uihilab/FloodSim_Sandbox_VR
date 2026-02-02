// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SimConfiguration.generated.h"

UCLASS()
class VR_VIS_API ASimConfiguration : public AActor
{
	GENERATED_BODY()

protected:
	virtual void BeginPlay() override;
	void ReadFiles();

public:	
	ASimConfiguration();
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Files")
	TArray<FString> FileNames{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Files")
	FString SelectedName{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Files")
	FString ShortName{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool UsingVR{false};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool NPCConfig{true};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Configuration")
	bool VehicleConfig{true};

	UFUNCTION(BlueprintCallable, Category = "Files")
	bool SelectFiles(FString& OutFileName);

	UFUNCTION(BlueprintPure, Category = "Files")
	FString GetFullPath() const;

	UFUNCTION(BlueprintCallable, Category = "Files")
	void SubmitConfig();
};
