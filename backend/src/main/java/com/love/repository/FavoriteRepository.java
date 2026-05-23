package com.love.repository;

import com.love.entity.Favorite;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface FavoriteRepository extends JpaRepository<Favorite, Long> {
    List<Favorite> findByUserId(Long userId);
    Optional<Favorite> findByUserIdAndMusicId(Long userId, Long musicId);
    boolean existsByUserIdAndMusicId(Long userId, Long musicId);
}
