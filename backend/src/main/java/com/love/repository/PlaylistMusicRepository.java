package com.love.repository;

import com.love.entity.PlaylistMusic;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PlaylistMusicRepository extends JpaRepository<PlaylistMusic, Long> {
    List<PlaylistMusic> findByPlaylistIdOrderBySortOrder(Long playlistId);
    void deleteByPlaylistIdAndMusicId(Long playlistId, Long musicId);
}
