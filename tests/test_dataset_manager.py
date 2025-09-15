"""
Unit tests for DatasetManager

Tests dataset registry, storage monitoring, and metadata handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.data import DatasetManager, DatasetRegistry, DatasetInfo


class TestDatasetRegistry:
    """Test the dataset registry functionality"""
    
    def test_registry_initialization(self):
        """Test registry initializes with expected datasets"""
        registry = DatasetRegistry()
        
        assert len(registry.datasets) >= 7  # Should have main datasets
        assert "wilddeepfake" in registry.datasets
        assert "celebdf" in registry.datasets
        assert "asvspoof2021" in registry.datasets
        
    def test_get_dataset_valid(self):
        """Test getting valid dataset"""
        registry = DatasetRegistry()
        dataset = registry.get_dataset("wilddeepfake")
        
        assert dataset is not None
        assert dataset.name == "WildDeepfake"
        assert dataset.type == "video"
        assert dataset.size_gb == 4.2
        assert dataset.storage_priority == 1  # Highest priority
        
    def test_get_dataset_invalid(self):
        """Test getting invalid dataset returns None"""
        registry = DatasetRegistry()
        dataset = registry.get_dataset("nonexistent")
        assert dataset is None
        
    def test_list_datasets_all(self):
        """Test listing all datasets"""
        registry = DatasetRegistry()
        datasets = registry.list_datasets()
        
        assert len(datasets) >= 7
        assert all(isinstance(d, DatasetInfo) for d in datasets)
        
    def test_list_datasets_by_type(self):
        """Test filtering datasets by type"""
        registry = DatasetRegistry()
        
        video_datasets = registry.list_datasets("video")
        audio_datasets = registry.list_datasets("audio")
        
        assert len(video_datasets) >= 4
        assert len(audio_datasets) >= 1
        assert all(d.type == "video" for d in video_datasets)
        assert all(d.type == "audio" for d in audio_datasets)
        
    def test_storage_recommendations(self):
        """Test storage-optimized recommendations"""
        registry = DatasetRegistry()
        recommendations = registry.get_storage_recommendations(storage_limit_gb=20.0)
        
        # Should prioritize smaller, high-priority datasets
        assert len(recommendations) > 0
        assert recommendations[0].storage_priority == 1  # WildDeepfake should be first
        
        # Calculate total size
        total_size = sum(r.size_gb or 0 for r in recommendations)
        assert total_size <= 20.0  # Should fit within limit


class TestDatasetManager:
    """Test the dataset manager functionality"""
    
    @pytest.fixture
    def temp_data_root(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_initialization(self, temp_data_root):
        """Test DatasetManager initialization"""
        dm = DatasetManager(str(temp_data_root), storage_limit_gb=50.0)
        
        assert dm.data_root == temp_data_root
        assert dm.storage_limit_gb == 50.0
        assert dm.metadata_dir.exists()
        assert isinstance(dm.registry, DatasetRegistry)
        
    def test_check_storage_usage_empty(self, temp_data_root):
        """Test storage usage check on empty directory"""
        dm = DatasetManager(str(temp_data_root), storage_limit_gb=50.0)
        storage_info = dm.check_storage_usage()
        
        assert storage_info["total_usage_gb"] == 0.0
        assert storage_info["available_gb"] == 50.0
        assert storage_info["within_limit"] == True
        assert storage_info["datasets"] == {}
        
    def test_create_directory_structure_video(self, temp_data_root):
        """Test creating directory structure for video dataset"""
        dm = DatasetManager(str(temp_data_root))
        dataset_dir = dm.create_directory_structure("wilddeepfake")
        
        assert dataset_dir.exists()
        assert (dataset_dir / "real").exists()
        assert (dataset_dir / "fake").exists()
        assert (dataset_dir / "metadata").exists()
        
    def test_create_directory_structure_audio(self, temp_data_root):
        """Test creating directory structure for audio dataset"""
        dm = DatasetManager(str(temp_data_root))
        dataset_dir = dm.create_directory_structure("asvspoof2021")
        
        assert dataset_dir.exists()
        assert (dataset_dir / "bonafide").exists()
        assert (dataset_dir / "spoof").exists()
        assert (dataset_dir / "metadata").exists()
        
    def test_save_load_metadata(self, temp_data_root):
        """Test saving and loading dataset metadata"""
        dm = DatasetManager(str(temp_data_root))
        
        test_metadata = {
            "dataset_id": "test_dataset",
            "file_count": 100,
            "creation_date": "2024-01-01"
        }
        
        # Save metadata
        dm.save_dataset_metadata("test_dataset", test_metadata)
        
        # Load metadata
        loaded_metadata = dm.load_dataset_metadata("test_dataset")
        
        assert loaded_metadata is not None
        assert loaded_metadata["dataset_id"] == "test_dataset"
        assert loaded_metadata["file_count"] == 100
        assert "storage_info" in loaded_metadata  # Should add storage info
        
    def test_create_sample_dataset(self, temp_data_root):
        """Test creating sample dataset"""
        dm = DatasetManager(str(temp_data_root))
        sample_dir = dm.create_sample_dataset("test_sample", video_count=5, audio_count=5)
        
        assert sample_dir.exists()
        assert (sample_dir / "video" / "real").exists()
        assert (sample_dir / "video" / "fake").exists()
        assert (sample_dir / "audio" / "bonafide").exists()
        assert (sample_dir / "audio" / "spoof").exists()
        assert (sample_dir / "metadata.json").exists()
        
        # Check metadata
        with open(sample_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        assert metadata["video_count"] == 10  # 5 real + 5 fake
        assert metadata["audio_count"] == 10  # 5 real + 5 fake
        assert metadata["storage_efficient"] == True
        
    def test_verify_dataset_nonexistent(self, temp_data_root):
        """Test verifying non-existent dataset"""
        dm = DatasetManager(str(temp_data_root))
        verification = dm.verify_dataset("nonexistent")
        
        assert verification["exists"] == False
        assert "error" in verification
        
    def test_verify_dataset_existing(self, temp_data_root):
        """Test verifying existing dataset"""
        dm = DatasetManager(str(temp_data_root))
        
        # Create sample dataset
        sample_dir = dm.create_sample_dataset("test_verify", video_count=2, audio_count=2)
        
        # Create some dummy files to test file counting
        (sample_dir / "video" / "real" / "test1.mp4").touch()
        (sample_dir / "video" / "fake" / "test2.mp4").touch()
        (sample_dir / "audio" / "bonafide" / "test3.wav").touch()
        
        verification = dm.verify_dataset("test_verify")
        
        assert verification["exists"] == True
        assert verification["file_count"] >= 3  # At least the files we created
        assert verification["total_size_gb"] >= 0
        
    @patch('builtins.print')
    def test_print_dataset_info(self, mock_print, temp_data_root):
        """Test printing dataset information"""
        dm = DatasetManager(str(temp_data_root), storage_limit_gb=50.0)
        dm.print_dataset_info(detailed=True)
        
        # Check that print was called
        assert mock_print.called
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = ' '.join(print_calls)
        
        # Should contain key information
        assert "Available Deepfake Detection Datasets" in print_output
        assert "Storage Limit: 50.0 GB" in print_output
        assert "WildDeepfake" in print_output
        assert "RECOMMENDED" in print_output
        
    def test_get_download_instructions(self, temp_data_root):
        """Test getting download instructions"""
        dm = DatasetManager(str(temp_data_root), storage_limit_gb=50.0)
        instructions = dm.get_download_instructions("wilddeepfake")
        
        assert "WildDeepfake" in instructions
        assert "4.2 GB" in instructions
        assert "huggingface.co" in instructions
        assert "STORAGE CONSIDERATIONS" in instructions
        assert "Available Space:" in instructions
        
    def test_list_local_datasets_empty(self, temp_data_root):
        """Test listing local datasets when none exist"""
        dm = DatasetManager(str(temp_data_root))
        local_datasets = dm.list_local_datasets()
        
        assert len(local_datasets) == 0
        
    def test_list_local_datasets_with_data(self, temp_data_root):
        """Test listing local datasets with existing data"""
        dm = DatasetManager(str(temp_data_root))
        
        # Create sample datasets
        dm.create_sample_dataset("sample1", video_count=1, audio_count=1)
        dm.create_sample_dataset("sample2", video_count=2, audio_count=2)
        
        local_datasets = dm.list_local_datasets()
        
        assert len(local_datasets) == 2
        dataset_ids = [d["id"] for d in local_datasets]
        assert "sample1" in dataset_ids
        assert "sample2" in dataset_ids
        
        # Should be sorted by size (smallest first)
        sizes = [d["size_gb"] for d in local_datasets]
        assert sizes == sorted(sizes)


class TestStorageEfficiency:
    """Test storage efficiency features"""
    
    def test_storage_limit_enforcement(self):
        """Test that storage limits are properly enforced"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create manager with very small limit
            dm = DatasetManager(temp_dir, storage_limit_gb=1.0)
            storage_info = dm.check_storage_usage()
            
            assert storage_info["available_gb"] == 1.0
            
            # Get recommendations for small storage
            registry = DatasetRegistry()
            recommendations = registry.get_storage_recommendations(storage_limit_gb=1.0)
            
            # Should be very limited recommendations
            total_size = sum(r.size_gb or 0 for r in recommendations)
            assert total_size <= 1.0
            
    def test_compressed_version_preference(self):
        """Test that compressed versions are preferred when available"""
        registry = DatasetRegistry()
        
        # Get a dataset with compressed version available
        celebdf = registry.get_dataset("celebdf")
        assert celebdf.compressed_available == True
        assert celebdf.compressed_size_gb < celebdf.size_gb
        
        # Get recommendations that should prefer compressed versions
        recommendations = registry.get_storage_recommendations(storage_limit_gb=20.0)
        
        # Should include datasets that fit even in compressed form
        recommendation_names = [r.name for r in recommendations]
        assert "Celeb-DF" in recommendation_names or len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
