package com.love.repository;

import com.love.entity.Music;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface MusicRepository extends JpaRepository<Music, Long> {
    @Query("SELECT m FROM Music m WHERE m.status = 1 AND (m.title LIKE %:keyword% OR m.artist LIKE %:keyword%)")
    Page<Music> searchByKeyword(@Param("keyword") String keyword, Pageable pageable);

    Page<Music> findByStatus(Byte status, Pageable pageable);
}
