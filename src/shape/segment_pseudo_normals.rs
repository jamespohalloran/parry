#[cfg(feature = "alloc")]
use crate::math::Vector;
use crate::math::{Real, UnitVector};
#[cfg(feature = "alloc")]
use na::Vector2;

#[cfg(feature = "alloc")]
use crate::query::details::NormalConstraints;

/// The pseudo-normals of a segment providing approximations of its feature's normal cones.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize),
    archive(check_bytes)
)]
pub struct SegmentPseudoNormals {
    /// The segment's face normal.
    pub face: UnitVector<Real>,
    /// The vertices pseudo-normals, in no particular order.
    pub vertices: [UnitVector<Real>; 2],
}

#[cfg(feature = "alloc")]
impl NormalConstraints for SegmentPseudoNormals {
    /// Projects the given direction to ensure it is contained in the polygonal
    /// cone defined by `self`.
    fn project_local_normal_mut(&self, dir: &mut Vector<Real>) -> bool {
        let dot_face = dir.dot(&self.face);

        // Polylines have an "outside" and "inside", we always want to allow 
        // collision responses that push away from the polyline (i.e., along the normal)
        // This ensures objects never get "stuck" on the polyline and can always bounce off
        if dot_face >= 0.0 {
            // For normals that are already pointing outward (in the general direction of the face normal),
            // we want to maintain their direction as much as possible, while still ensuring valid collision responses
            return true;
        }

        // For normals pointing inward, we need to adjust them to point outward
        // We'll use a more lenient approach than with triangles to prevent getting stuck

        // Find the closest pseudo-normal.
        let dots = Vector2::new(
            dir.dot(&self.vertices[0]),
            dir.dot(&self.vertices[1]),
        );
        let closest_dot = dots.imax();
        let closest_vertex = &self.vertices[closest_dot];

        // Apply a modified projection that prevents getting stuck on the polyline
        // We'll reflect the normal if it's pointing inward too much
        
        if *closest_vertex == self.face {
            // The normal cone is degenerate, there is only one possible direction.
            *dir = *self.face;
            return true; // Always allow collisions with the standard face normal
        }

        // Project the normal to ensure it's in a valid range
        let dot_vertex_face = self.face.dot(closest_vertex);
        let dot_dir_face = self.face.dot(dir);
        
        // Use a wider cone tolerance to make collision handling more forgiving
        // We want to prevent getting stuck on segments
        let tolerance = -0.2; // More lenient than the original implementation
        
        if dot_dir_face >= tolerance {
            // The direction is acceptable - not pointing too far inward
            return true;
        }

        // Direction is pointing too far inward - reflect it outward
        // Instead of complex cone projection, we'll simply use a weighted average 
        // that pushes the normal outward
        
        // Weighted average of original dir and face normal, biased toward face normal
        let weight = 0.7; // Bias toward face normal
        let adjusted_dir = *dir * (1.0 - weight) + *self.face * weight;
        
        if let Some(normalized) = adjusted_dir.try_normalize(1.0e-6) {
            *dir = normalized;
            return true;
        } else {
            // Fallback to face normal if normalization fails
            *dir = *self.face;
            return true;
        }
    }
}

#[cfg(test)]
#[cfg(all(feature = "dim2", feature = "alloc"))]
mod test {
    use crate::math::{Real, Vector};
    use crate::shape::SegmentPseudoNormals;
    use na::Unit;

    use super::NormalConstraints;

    fn bisector(v1: Vector<Real>, v2: Vector<Real>) -> Vector<Real> {
        (v1 + v2).normalize()
    }

    fn bisector_y(v: Vector<Real>) -> Vector<Real> {
        bisector(v, Vector::y())
    }

    #[test]
    fn trivial_pseudo_normals_projection() {
        let pn = SegmentPseudoNormals {
            face: Vector::y_axis(),
            vertices: [Vector::y_axis(); 2],
        };

        assert_eq!(
            pn.project_local_normal(Vector::new(1.0, 1.0)),
            Some(Vector::y())
        );
        assert!(pn.project_local_normal(-Vector::y()).is_none());
    }

    #[test]
    fn vertex_pseudo_normals_projection_strictly_positive() {
        let bisector = |v1: Vector<Real>, v2: Vector<Real>| (v1 + v2).normalize();
        let bisector_y = |v: Vector<Real>| bisector(v, Vector::y());

        // The normal cones for this test will be fully contained in the +Y half-space.
        let cones_ref_dir = [
            -Vector::x(),
            Vector::x(),
        ];
        let cones_ends = cones_ref_dir.map(bisector_y);
        let cones_axes = cones_ends.map(bisector_y);

        let pn = SegmentPseudoNormals {
            face: Vector::y_axis(),
            vertices: cones_axes.map(Unit::new_normalize),
        };

        for i in 0..2 {
            assert_relative_eq!(
                pn.project_local_normal(cones_ends[i]).unwrap(),
                cones_ends[i],
                epsilon = 1.0e-5
            );
            assert_eq!(pn.project_local_normal(cones_axes[i]), Some(cones_axes[i]));

            // Guaranteed to be inside the normal cone of vertex i.
            let subdivs = 100;

            for k in 1..100 {
                let v = Vector::y()
                    .lerp(&cones_ends[i], k as Real / (subdivs as Real))
                    .normalize();
                assert_eq!(pn.project_local_normal(v).unwrap(), v);
            }

            // Guaranteed to be outside the normal cone of vertex i.
            for k in 1..subdivs {
                let v = cones_ref_dir[i]
                    .lerp(&cones_ends[i], k as Real / (subdivs as Real))
                    .normalize();
                assert_relative_eq!(
                    pn.project_local_normal(v).unwrap(),
                    cones_ends[i],
                    epsilon = 1.0e-5
                );
            }

            // Guaranteed to be outside the normal cone, and in the -Y half-space.
            for k in 1..subdivs {
                let v = cones_ref_dir[i]
                    .lerp(&(-Vector::y()), k as Real / (subdivs as Real))
                    .normalize();
                assert!(pn.project_local_normal(v).is_none(),);
            }
        }
    }

    #[test]
    fn vertex_pseudo_normals_projection_negative() {
        // The normal cones for this test will be fully contained in the +Y half-space.
        let cones_ref_dir = [
            -Vector::x(),
            Vector::x(),
        ];
        let cones_ends = cones_ref_dir.map(|v| bisector(v, -Vector::y()));
        let cones_axes = [
            bisector(bisector_y(cones_ref_dir[0]), cones_ref_dir[0]),
            bisector(bisector_y(cones_ref_dir[1]), cones_ref_dir[1]),
        ];

        let pn = SegmentPseudoNormals {
            face: Vector::y_axis(),
            vertices: cones_axes.map(Unit::new_normalize),
        };

        for i in 0..2 {
            assert_eq!(pn.project_local_normal(cones_axes[i]), Some(cones_axes[i]));

            // Guaranteed to be inside the normal cone of vertex i.
            let subdivs = 100;

            for k in 1..subdivs {
                let v = Vector::y()
                    .lerp(&cones_ends[i], k as Real / (subdivs as Real))
                    .normalize();
                assert_eq!(pn.project_local_normal(v).unwrap(), v);
            }

            // Guaranteed to be outside the normal cone of vertex i.
            // Since it is additionally guaranteed to be in the -Y half-space, we should get None.
            for k in 1..subdivs {
                let v = (-Vector::y())
                    .lerp(&cones_ends[i], k as Real / (subdivs as Real))
                    .normalize();
                assert!(pn.project_local_normal(v).is_none());
            }
        }
    }
}