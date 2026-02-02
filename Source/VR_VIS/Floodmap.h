// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"
#include "http.h"
#include "GeoReferencingSystem.h"
#include "Materials/MaterialInterface.h"
#include "Floodmap.generated.h"

UCLASS()
class VR_VIS_API AFloodmap : public AActor
{
	GENERATED_BODY()

protected:
	UMaterialInstanceDynamic* DynamicMat{};
	TArray<int32> MeshSectionIndices{};
	int32 CurrentMeshSectionIndex{};

	virtual void BeginPlay() override;
	void MakeHTTPRequest();
	void OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool wasSuccessful);
	void ReadFile();
	void ParseFile(const FString& FileContent);
	FVector LatLongToWorldspace(double longitude, double latitude, double altitude);
	void UpdateMaterial();

public:	
	AFloodmap();
	virtual void Tick(float DeltaTime) override;
	void DrawFloodMap(const TArray<FVector>& Vertices);

	// Components & Variables

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UProceduralMeshComponent* ProceduralMeshComponent{};

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Georeferencing")
	AGeoReferencingSystem* GeoReferencingSystem{};

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Materials")
	UMaterialInterface* Material{};

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Materials")
	TArray<UMaterialInstanceDynamic*> InstancedMaterials{};
};
