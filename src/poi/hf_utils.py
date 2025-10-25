"""
Hugging Face Utilities

This module provides utility functions for interacting with Hugging Face Hub,
including listing models, checking model details, and filtering models by type.
"""

import os
from typing import List, Optional, Dict, Any
from huggingface_hub import HfApi, ModelInfo
from .settings import HF_ORG, HF_TOKEN


def get_hf_api() -> HfApi:
    """
    Get a configured Hugging Face API client.
    
    Returns:
        HfApi: Configured Hugging Face API client
        
    Raises:
        ValueError: If HF_TOKEN is not set
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
    
    return HfApi(token=HF_TOKEN)


def list_all_models(organization: Optional[str] = None) -> List[ModelInfo]:
    """
    List all models in the specified organization or all accessible models.
    
    Args:
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
        
    Returns:
        List[ModelInfo]: List of model information objects
        
    Raises:
        Exception: If there's an error fetching models from Hugging Face
    """
    api = get_hf_api()
    org = organization or HF_ORG
    
    try:
        models = list(api.list_models(author=org))
        return models
    except Exception as e:
        raise Exception(f"Error fetching models from organization '{org}': {e}")


def filter_models_by_keywords(models: List[ModelInfo], keywords: List[str]) -> List[ModelInfo]:
    """
    Filter models by keywords in their model ID.
    
    Args:
        models: List of model information objects
        keywords: List of keywords to search for in model IDs
        
    Returns:
        List[ModelInfo]: Filtered list of models containing any of the keywords
    """
    filtered_models = []
    
    for model in models:
        model_id_lower = model.id.lower()
        if any(keyword.lower() in model_id_lower for keyword in keywords):
            filtered_models.append(model)
    
    return filtered_models


def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Full model ID (e.g., 'organization/model-name')
        
    Returns:
        Optional[Dict[str, Any]]: Model details dictionary, or None if model not found
        
    Raises:
        Exception: If there's an error fetching model details
    """
    api = get_hf_api()
    
    try:
        model_info = api.model_info(model_id)
        
        # Get model files
        try:
            files = api.list_repo_files(model_id)
        except Exception as e:
            files = [f"Error listing files: {e}"]
        
        return {
            'id': model_info.id,
            'created_at': model_info.created_at,
            'downloads': model_info.downloads,
            'tags': model_info.tags or [],
            'pipeline_tag': model_info.pipeline_tag,
            'files': files,
            'card_data': getattr(model_info, 'card_data', None),
            'safetensors': getattr(model_info, 'safetensors', None),
            'library_name': getattr(model_info, 'library_name', None),
        }
        
    except Exception as e:
        raise Exception(f"Error fetching model details for '{model_id}': {e}")


def list_model_files(model_id: str) -> List[str]:
    """
    List all files in a model repository.
    
    Args:
        model_id: Full model ID (e.g., 'organization/model-name')
        
    Returns:
        List[str]: List of file paths in the repository
        
    Raises:
        Exception: If there's an error listing files
    """
    api = get_hf_api()
    
    try:
        files = api.list_repo_files(model_id)
        return files
    except Exception as e:
        raise Exception(f"Error listing files for model '{model_id}': {e}")


def check_rqvae_models(organization: Optional[str] = None) -> List[ModelInfo]:
    """
    Find all RQ-VAE related models in the organization.
    
    Args:
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
        
    Returns:
        List[ModelInfo]: List of RQ-VAE related models
    """
    all_models = list_all_models(organization)
    rqvae_keywords = ['rqvae', 'rq-vae', 'rq_vae', 'vae']
    return filter_models_by_keywords(all_models, rqvae_keywords)


def check_llm_models(organization: Optional[str] = None) -> List[ModelInfo]:
    """
    Find all LLM related models in the organization.
    
    Args:
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
        
    Returns:
        List[ModelInfo]: List of LLM related models
    """
    all_models = list_all_models(organization)
    llm_keywords = ['llm', 'llama', 'gpt', 'bert', 'roberta', 't5', 'transformer']
    return filter_models_by_keywords(all_models, llm_keywords)


def print_model_summary(models: List[ModelInfo], title: str = "Models") -> None:
    """
    Print a formatted summary of models.
    
    Args:
        models: List of model information objects
        title: Title for the summary
    """
    print(f"📋 {title} Summary:")
    print(f"   Total models found: {len(models)}")
    
    if models:
        print(f"\n📝 Model IDs:")
        for i, model in enumerate(models, 1):
            print(f"   {i:2d}. {model.id}")


def print_model_details(model_details: Dict[str, Any]) -> None:
    """
    Print formatted model details.
    
    Args:
        model_details: Dictionary containing model information
    """
    print(f"✅ Model: {model_details['id']}")
    print(f"   - Created: {model_details['created_at']}")
    print(f"   - Downloads: {model_details['downloads']}")
    print(f"   - Tags: {', '.join(model_details['tags']) if model_details['tags'] else 'None'}")
    print(f"   - Pipeline Tag: {model_details['pipeline_tag'] or 'None'}")
    print(f"   - Library: {model_details['library_name'] or 'None'}")
    
    print(f"\n📁 Model files ({len(model_details['files'])} files):")
    for file in model_details['files'][:10]:  # Show first 10 files
        print(f"   - {file}")
    
    if len(model_details['files']) > 10:
        print(f"   ... and {len(model_details['files']) - 10} more files")


def check_all_models(organization: Optional[str] = None) -> List[ModelInfo]:
    """
    Check and display all models in the organization.
    
    Args:
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
        
    Returns:
        List[ModelInfo]: List of all models found
    """
    org = organization or HF_ORG
    print(f"🔍 Checking ALL Hugging Face Models in Organization: {org}")
    print("=" * 60)
    
    try:
        models = list_all_models(organization)
        print(f"✅ Found {len(models)} models in organization '{org}':")
        print("-" * 60)
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. Model ID: {model.id}")
            print(f"    Created: {model.created_at}")
            print(f"    Downloads: {model.downloads}")
            print(f"    Tags: {', '.join(model.tags) if model.tags else 'None'}")
            print(f"    Pipeline Tag: {model.pipeline_tag if model.pipeline_tag else 'None'}")
            print()
        
        return models
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def check_specific_model(model_id: str) -> None:
    """
    Check details of a specific model.
    
    Args:
        model_id: Full model ID (e.g., 'organization/model-name')
    """
    print(f"🔍 Checking model details: {model_id}")
    print("=" * 50)
    
    try:
        model_details = get_model_details(model_id)
        print_model_details(model_details)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def check_filtered_models(filter_type: str, organization: Optional[str] = None) -> List[ModelInfo]:
    """
    Check models filtered by type.
    
    Args:
        filter_type: Type of models to filter ('rqvae' or 'llm')
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
        
    Returns:
        List[ModelInfo]: List of filtered models
    """
    print(f"🔍 Checking {filter_type.upper()} related models...")
    print("=" * 50)
    
    try:
        if filter_type.lower() == 'rqvae':
            models = check_rqvae_models(organization)
            print(f"✅ Found {len(models)} RQ-VAE related models:")
        elif filter_type.lower() == 'llm':
            models = check_llm_models(organization)
            print(f"✅ Found {len(models)} LLM related models:")
        else:
            print(f"❌ Unknown filter type: {filter_type}")
            print("Available filter types: 'rqvae', 'llm'")
            return []
        
        if models:
            for model in models:
                print(f"  - {model.id}")
        else:
            print(f"❌ No {filter_type.upper()} related models found")
        
        return models
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def run_model_checker(model_id: Optional[str] = None, 
                     filter_type: Optional[str] = None, 
                     summary_only: bool = False,
                     organization: Optional[str] = None) -> None:
    """
    Main function to run the model checker with various options.
    
    Args:
        model_id: Specific model ID to check details for
        filter_type: Type of models to filter ('rqvae' or 'llm')
        summary_only: Whether to show only summary information
        organization: Hugging Face organization name. If None, uses HF_ORG from settings.
    """
    print("🚀 Hugging Face Models Checker")
    print("=" * 60)
    
    # Check if HF_TOKEN is available
    try:
        get_hf_api()
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 Please set your HF_TOKEN environment variable")
        return
    
    # Execute based on arguments
    if model_id:
        check_specific_model(model_id)
    elif filter_type:
        models = check_filtered_models(filter_type, organization)
        if summary_only and models:
            print_model_summary(models, f"{filter_type.upper()} Models")
    else:
        models = check_all_models(organization)
        if summary_only and models:
            print_model_summary(models, "All Models")
        
        # If we have models and no specific filter, show RQ-VAE models as well
        if models and not summary_only:
            print("\n" + "=" * 60)
            rqvae_models = check_rqvae_models(organization)
            if rqvae_models:
                print(f"\n📋 RQ-VAE Models Summary:")
                print(f"   Found {len(rqvae_models)} RQ-VAE related models")
                for model in rqvae_models:
                    print(f"   - {model.id}")
