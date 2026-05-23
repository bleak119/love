package com.love.controller;

import com.love.dto.ApiResponse;
import com.love.entity.Playlist;
import com.love.entity.PlaylistMusic;
import com.love.service.PlaylistService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@Slf4j
@RestController
@RequestMapping("/playlists")
public class PlaylistController {
    private final PlaylistService playlistService;

    public PlaylistController(PlaylistService playlistService) {
        this.playlistService = playlistService;
    }

    @PostMapping
    public ResponseEntity<ApiResponse<?>> createPlaylist(@RequestParam String name) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            Playlist playlist = playlistService.createPlaylist(userId, name);
            return ResponseEntity.ok(ApiResponse.success(playlist));
        } catch (Exception e) {
            log.error("Create playlist failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to create playlist"));
        }
    }

    @GetMapping
    public ResponseEntity<ApiResponse<?>> getUserPlaylists() {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            List<Playlist> playlists = playlistService.getUserPlaylists(userId);
            return ResponseEntity.ok(ApiResponse.success(playlists));
        } catch (Exception e) {
            log.error("Get user playlists failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to get playlists"));
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<?>> getPlaylist(@PathVariable Long id) {
        try {
            Optional<Playlist> playlist = playlistService.getPlaylist(id);
            if (playlist.isEmpty()) {
                return ResponseEntity.status(404)
                        .body(ApiResponse.error(404, "Playlist not found"));
            }
            return ResponseEntity.ok(ApiResponse.success(playlist.get()));
        } catch (Exception e) {
            log.error("Get playlist failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to get playlist"));
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<?>> deletePlaylist(@PathVariable Long id) {
        try {
            playlistService.deletePlaylist(id);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Delete playlist failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to delete playlist"));
        }
    }

    @PostMapping("/{id}/music/{musicId}")
    public ResponseEntity<ApiResponse<?>> addMusicToPlaylist(
            @PathVariable Long id,
            @PathVariable Long musicId,
            @RequestParam(required = false) Integer sortOrder) {
        try {
            playlistService.addMusicToPlaylist(id, musicId, sortOrder);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Add music to playlist failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to add music to playlist"));
        }
    }

    @DeleteMapping("/{id}/music/{musicId}")
    public ResponseEntity<ApiResponse<?>> removeMusicFromPlaylist(
            @PathVariable Long id,
            @PathVariable Long musicId) {
        try {
            playlistService.removeMusicFromPlaylist(id, musicId);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Remove music from playlist failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to remove music from playlist"));
        }
    }

    @GetMapping("/{id}/music")
    public ResponseEntity<ApiResponse<?>> getPlaylistMusics(@PathVariable Long id) {
        try {
            List<PlaylistMusic> musics = playlistService.getPlaylistMusics(id);
            return ResponseEntity.ok(ApiResponse.success(musics));
        } catch (Exception e) {
            log.error("Get playlist musics failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to get playlist musics"));
        }
    }
}
