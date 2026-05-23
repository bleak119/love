package com.love.controller;

import com.love.dto.ApiResponse;
import com.love.entity.Favorite;
import com.love.service.FavoriteService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/favorites")
public class FavoriteController {
    private final FavoriteService favoriteService;

    public FavoriteController(FavoriteService favoriteService) {
        this.favoriteService = favoriteService;
    }

    @PostMapping("/{musicId}")
    public ResponseEntity<ApiResponse<?>> addFavorite(@PathVariable Long musicId) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            favoriteService.addFavorite(userId, musicId);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Add favorite failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to add favorite"));
        }
    }

    @DeleteMapping("/{musicId}")
    public ResponseEntity<ApiResponse<?>> removeFavorite(@PathVariable Long musicId) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            favoriteService.removeFavorite(userId, musicId);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Remove favorite failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to remove favorite"));
        }
    }

    @GetMapping
    public ResponseEntity<ApiResponse<?>> getFavorites() {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            List<Favorite> favorites = favoriteService.getUserFavorites(userId);
            return ResponseEntity.ok(ApiResponse.success(favorites));
        } catch (Exception e) {
            log.error("Get favorites failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to get favorites"));
        }
    }

    @GetMapping("/{musicId}/check")
    public ResponseEntity<ApiResponse<?>> checkFavorite(@PathVariable Long musicId) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            boolean isFavorite = favoriteService.isFavorite(userId, musicId);
            return ResponseEntity.ok(ApiResponse.success(isFavorite));
        } catch (Exception e) {
            log.error("Check favorite failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to check favorite"));
        }
    }
}
