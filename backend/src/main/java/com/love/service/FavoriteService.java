package com.love.service;

import com.love.entity.Favorite;
import com.love.repository.FavoriteRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class FavoriteService {
    private final FavoriteRepository favoriteRepository;

    public FavoriteService(FavoriteRepository favoriteRepository) {
        this.favoriteRepository = favoriteRepository;
    }

    public void addFavorite(Long userId, Long musicId) {
        if (!favoriteRepository.existsByUserIdAndMusicId(userId, musicId)) {
            Favorite favorite = new Favorite();
            favorite.setUserId(userId);
            favorite.setMusicId(musicId);
            favoriteRepository.save(favorite);
        }
    }

    public void removeFavorite(Long userId, Long musicId) {
        favoriteRepository.findByUserIdAndMusicId(userId, musicId)
                .ifPresent(favoriteRepository::delete);
    }

    public List<Favorite> getUserFavorites(Long userId) {
        return favoriteRepository.findByUserId(userId);
    }

    public boolean isFavorite(Long userId, Long musicId) {
        return favoriteRepository.existsByUserIdAndMusicId(userId, musicId);
    }
}
