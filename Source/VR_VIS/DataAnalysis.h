// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/Texture2D.h"
#include "http.h"
#include "DataAnalysis.generated.h"

UCLASS()
class VR_VIS_API ADataAnalysis : public AActor
{
	GENERATED_BODY()

protected:
	virtual void BeginPlay() override;
	void MakeHTTPRequest(const FString& Base64String);
	void OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool wasSuccesful);

public:	
	ADataAnalysis();
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Camera")
	UCameraComponent* Camera{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Data Analysis")
	USceneCaptureComponent2D* SceneCaptureComponent{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Data Analysis")
	UTextureRenderTarget2D* RenderTarget{};

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Data Analysis")
	TArray<uint8> CompressedMap{};

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "UI")
	UTexture2D* Screenshot{};

	UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Data Analysis")
	FString GPTResponse{};

	UFUNCTION(BlueprintCallable, Category = "Data Analysis")
	void TakeScreenshot(const FString FileName);

	UFUNCTION(BlueprintCallable, Category = "Data Analysis")
	void AnalyzeScreenshot(const FString FileName);
};
