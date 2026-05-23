package com.love.service;

import com.love.entity.Playlist;
import com.love.entity.PlaylistMusic;
import com.love.repository.PlaylistRepository;
import com.love.repository.PlaylistMusicRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PlaylistService {
    private final PlaylistRepository playlistRepository;
    private final PlaylistMusicRepository playlistMusicRepository;

    public PlaylistService(PlaylistRepository playlistRepository, PlaylistMusicRepository playlistMusicRepository) {
        this.playlistRepository = playlistRepository;
        this.playlistMusicRepository = playlistMusicRepository;
    }

    public Playlist createPlaylist(Long userId, String name) {
        Playlist playlist = new Playlist();
        playlist.setUserId(userId);
        playlist.setName(name);
        playlist.setIsPublic((byte) 1);
        return playlistRepository.save(playlist);
    }

    public List<Playlist> getUserPlaylists(Long userId) {
        return playlistRepository.findByUserId(userId);
    }

    public Optional<Playlist> getPlaylist(Long playlistId) {
        return playlistRepository.findById(playlistId);
    }

    public void deletePlaylist(Long playlistId) {
        playlistRepository.deleteById(playlistId);
    }

    public void addMusicToPlaylist(Long playlistId, Long musicId, Integer sortOrder) {
        PlaylistMusic pm = new PlaylistMusic();
        pm.setPlaylistId(playlistId);
        pm.setMusicId(musicId);
        pm.setSortOrder(sortOrder != null ? sortOrder : 0);
        playlistMusicRepository.save(pm);
    }

    public void removeMusicFromPlaylist(Long playlistId, Long musicId) {
        playlistMusicRepository.deleteByPlaylistIdAndMusicId(playlistId, musicId);
    }

    public List<PlaylistMusic> getPlaylistMusics(Long playlistId) {
        return playlistMusicRepository.findByPlaylistIdOrderBySortOrder(playlistId);
    }
}
